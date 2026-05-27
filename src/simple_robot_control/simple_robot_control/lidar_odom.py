# Authors: Akash Mohapatra, Manas Chintawar

#!/usr/bin/env python3
"""
2D LiDAR Odometry - Scan-to-Local-Map ICP
Matches current scan against a rolling window of the last N keyframe scans.
This is more stable than a growing voxel map for 2D LiDAR.
"""

import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster


# ── SE(2) helpers ─────────────────────────────────────────────────────────────

def make_T(dx: float, dy: float, dth: float) -> np.ndarray:
    c, s = math.cos(dth), math.sin(dth)
    return np.array([[c, -s, dx],
                     [s,  c, dy],
                     [0,  0,  1]], dtype=np.float64)


def T_to_pose(T: np.ndarray):
    return T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:2, :2]
    t = T[:2, 2]
    Ti = np.eye(3)
    Ti[:2, :2] = R.T
    Ti[:2, 2]  = -R.T @ t
    return Ti


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:2, :2].T + T[:2, 2]


# ── Point cloud utils ─────────────────────────────────────────────────────────

def scan_to_points(msg: LaserScan, max_range: float) -> np.ndarray:
    angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
    r = np.asarray(msg.ranges, dtype=np.float64)
    valid = np.isfinite(r) & (r > msg.range_min) & (r < max_range)
    return np.column_stack([r[valid] * np.cos(angles[valid]),
                            r[valid] * np.sin(angles[valid])])


def voxel_downsample(pts: np.ndarray, vsize: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / vsize).astype(np.int32)
    packed = keys[:, 0].astype(np.int64) * (2 ** 20) + keys[:, 1].astype(np.int64)
    _, first = np.unique(packed, return_index=True)
    return pts[first]


# ── ICP (point-to-point, weighted Cauchy) ─────────────────────────────────────

def cauchy_weight(d2: np.ndarray, c: float) -> np.ndarray:
    return (c * c) / (c * c + d2)


def icp(src: np.ndarray, dst: np.ndarray,
        max_dist: float, max_iter: int = 30, tol: float = 1e-5):
    """
    Align src → dst. Returns (T_corr, rmse).
    T_corr is in the local frame of src (identity = no correction needed).
    """
    T_acc = np.eye(3)
    pts   = src.copy()
    prev_err = 1e9

    for _ in range(max_iter):
        diff   = pts[:, None, :] - dst[None, :, :]   # (N,M,2)
        d2     = (diff ** 2).sum(2)                   # (N,M)
        nn_idx = d2.argmin(1)
        nn_d2  = d2[np.arange(len(pts)), nn_idx]

        inlier = nn_d2 < max_dist ** 2
        if inlier.sum() < 6:
            break

        s = pts[inlier]
        d = dst[nn_idx[inlier]]
        w = cauchy_weight(nn_d2[inlier], c=max_dist * 0.5)
        W = w / w.sum()

        ms = (W[:, None] * s).sum(0)
        md = (W[:, None] * d).sum(0)
        M  = ((s - ms) * w[:, None]).T @ (d - md)
        U, _, Vt = np.linalg.svd(M)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = md - R @ ms

        T_step = np.eye(3)
        T_step[:2, :2] = R
        T_step[:2, 2]  = t

        pts   = apply_T(T_step, pts)
        T_acc = T_step @ T_acc

        err = float(np.sqrt(nn_d2[inlier].mean()))
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    inlier = (((pts[:, None, :] - dst[None, :, :]) ** 2).sum(2).min(1)) < max_dist ** 2
    rmse   = float(np.sqrt((((pts[inlier] - dst[((pts[:, None, :] - dst[None, :, :]) ** 2).sum(2).argmin(1)[inlier]]) ** 2).sum(1).mean()))) \
             if inlier.sum() >= 6 else 1e9
    return T_acc, rmse


# ── Node ──────────────────────────────────────────────────────────────────────

class LidarOdomNode(Node):

    def __init__(self):
        super().__init__('lidar_odom_node')

        self.declare_parameter('max_range',        8.0)
        self.declare_parameter('voxel_size',       0.15)   # downsample resolution
        self.declare_parameter('icp_max_dist',     0.5)    # correspondence distance
        self.declare_parameter('min_points',       10)
        self.declare_parameter('map_window',       6)      # how many keyframe scans to keep
        self.declare_parameter('keyframe_dist',    0.15)   # meters moved before new keyframe
        self.declare_parameter('keyframe_angle',   0.08)   # radians turned before new keyframe
        self.declare_parameter('odom_frame',       'odom')
        self.declare_parameter('base_frame',       'base_footprint')
        self.declare_parameter('publish_tf',       True)
        self.declare_parameter('cov_x',            0.02)
        self.declare_parameter('cov_y',            0.02)
        self.declare_parameter('cov_yaw',          0.01)
        self.declare_parameter('cov_vx',           0.05)
        self.declare_parameter('cov_vyaw',         0.03)

        p = self.get_parameter
        self.max_range      = p('max_range').value
        self.vsize          = p('voxel_size').value
        self.icp_max_dist   = p('icp_max_dist').value
        self.min_pts        = p('min_points').value
        self.map_window     = p('map_window').value
        self.kf_dist        = p('keyframe_dist').value
        self.kf_angle       = p('keyframe_angle').value
        self.odom_frame     = p('odom_frame').value
        self.base_frame     = p('base_frame').value
        self.publish_tf     = p('publish_tf').value
        self.cov_x          = p('cov_x').value
        self.cov_y          = p('cov_y').value
        self.cov_yaw        = p('cov_yaw').value
        self.cov_vx         = p('cov_vx').value
        self.cov_vyaw       = p('cov_vyaw').value

        # State
        self.pose_T         = np.eye(3)
        self.prev_stamp     = None
        self.last_kf_T      = None          # pose at last keyframe
        self.odom_T         = np.eye(3)     # latest wheel odom pose
        self.prev_odom_T    = np.eye(3)     # wheel odom pose at last scan

        # Rolling window of keyframe scans in their own sensor frames,
        # plus the world-frame pose at the time they were captured.
        # Each entry: (pts_sensor_frame, T_world_from_sensor)
        self.keyframes: deque = deque(maxlen=self.map_window)

        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(LaserScan, '/lidar', self._scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.pub = self.create_publisher(Odometry, '/lidar_odom', 10)
        self.get_logger().info('LiDAR-Odom node started.')

    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.odom_T = make_T(x, y, yaw)

    def _build_local_map(self) -> np.ndarray:
        """
        Combine all keyframe scans into a single point cloud in world frame,
        then downsample. This is the ICP target.
        """
        all_pts = []
        for pts_sensor, T_kf in self.keyframes:
            all_pts.append(apply_T(T_kf, pts_sensor))
        if not all_pts:
            return np.empty((0, 2))
        combined = np.vstack(all_pts)
        return voxel_downsample(combined, self.vsize)

    def _scan_cb(self, msg: LaserScan):
        pts_raw = scan_to_points(msg, self.max_range)
        if len(pts_raw) < self.min_pts:
            self.get_logger().warn(f'Scan too small: {len(pts_raw)} pts', throttle_duration_sec=2.0)
            return
        pts = voxel_downsample(pts_raw, self.vsize)
        stamp = msg.header.stamp

        # Bootstrap on first scan — must come before anything else
        if not self.keyframes:
            self.get_logger().info(f'Bootstrap: {len(pts)} pts, seeding map.')
            self.keyframes.append((pts, self.pose_T.copy()))
            self.last_kf_T  = self.pose_T.copy()
            self.prev_stamp = stamp
            self._publish(stamp, 0.0, 0.0, 0.0, rmse=0.05)
            return

        map_world = self._build_local_map()
        if len(map_world) < self.min_pts:
            self.get_logger().warn(f'Local map too small: {len(map_world)}', throttle_duration_sec=2.0)
            return

        # Use wheel odom delta as prediction — far more reliable than constant velocity
        odom_delta = inv_T(self.prev_odom_T) @ self.odom_T
        T_pred = self.pose_T @ odom_delta
        self.prev_odom_T = self.odom_T.copy()
        pts_world_pred = apply_T(T_pred, pts)
        T_corr, rmse   = icp(pts_world_pred, map_world, max_dist=self.icp_max_dist)

        dx_c, dy_c, dth_c = T_to_pose(T_corr)
        trans_corr = math.hypot(dx_c, dy_c)
        icp_ok = not (trans_corr > 0.5 or abs(dth_c) > math.radians(30))
        if icp_ok:
            T_new = T_corr @ T_pred
        else:
            self.get_logger().warn(
                f'ICP too large: t={trans_corr:.3f}m r={math.degrees(dth_c):.1f}deg — using prediction')
            T_new = T_pred

        vx, vy, wz = 0.0, 0.0, 0.0
        if self.prev_stamp is not None:
            dt = (stamp.sec - self.prev_stamp.sec) + \
                (stamp.nanosec - self.prev_stamp.nanosec) * 1e-9
            if dt > 1e-6:
                dx_w, dy_w, dth_w = T_to_pose(odom_delta)
                c, s = T_new[0, 0], T_new[1, 0]
                vx   = ( dx_w * c + dy_w * s) / dt
                vy   = (-dx_w * s + dy_w * c) / dt
                wz   = dth_w / dt

        self.prev_stamp = stamp
        self.pose_T     = T_new

        # Only add keyframe when ICP succeeded — don't corrupt map with bad poses
        if icp_ok:
            dx_kf, dy_kf, dth_kf = T_to_pose(inv_T(self.last_kf_T) @ self.pose_T)
            if math.hypot(dx_kf, dy_kf) > self.kf_dist or abs(dth_kf) > self.kf_angle:
                self.keyframes.append((pts, self.pose_T.copy()))
                self.last_kf_T = self.pose_T.copy()

        self._publish(stamp, vx, vy, wz, rmse if icp_ok else 5.0)

    def _publish(self, stamp, vx, vy, wz, rmse=0.05):
        x, y, yaw = T_to_pose(self.pose_T)
        LARGE = 1e9

        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id  = self.base_frame

        odom.pose.pose.position.x    = float(x)
        odom.pose.pose.position.y    = float(y)
        odom.pose.pose.orientation.z = math.sin(yaw / 2)
        odom.pose.pose.orientation.w = math.cos(yaw / 2)

        pc = [0.0] * 36
        scale  = max(1.0, rmse / 0.05)
        pc[0]  = self.cov_x   * scale
        pc[7]  = self.cov_y   * scale
        pc[14] = LARGE
        pc[21] = LARGE
        pc[28] = LARGE
        pc[35] = self.cov_yaw * scale
        odom.pose.covariance = pc

        odom.twist.twist.linear.x  = float(vx)
        odom.twist.twist.linear.y  = float(vy)
        odom.twist.twist.angular.z = float(wz)

        tc = [0.0] * 36
        tc[0]  = self.cov_vx
        tc[7]  = LARGE
        tc[14] = LARGE
        tc[21] = LARGE
        tc[28] = LARGE
        tc[35] = self.cov_vyaw
        odom.twist.covariance = tc

        self.pub.publish(odom)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp       = stamp
            tf.header.frame_id    = self.odom_frame
            tf.child_frame_id     = self.base_frame
            tf.transform.translation.x = float(x)
            tf.transform.translation.y = float(y)
            tf.transform.rotation.z    = math.sin(yaw / 2)
            tf.transform.rotation.w    = math.cos(yaw / 2)
            self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = LidarOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()