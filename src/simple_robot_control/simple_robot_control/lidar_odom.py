# Authors: Akash Mohapatra, Manas Chintawar

#!/usr/bin/env python3
"""
2D LiDAR Odometry - Scan-to-Local-Map ICP
Matches current scan against a rolling window of the last N keyframe scans.
This is more stable than a growing voxel map for 2D LiDAR.
"""

import math
from collections import deque

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
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
    angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
    r = np.asarray(msg.ranges, dtype=np.float64)
    valid = (np.isfinite(r) & (r >= msg.range_min) & (r > 0.0)
             & (r <= min(msg.range_max, max_range)))
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


def _nearest_neighbors(src, dst):
    # Exact native nearest neighbors avoid allocating an N x M x 2 tensor
    # during every ICP iteration. Keep the registration algebra in float64.
    distances, indices = cv2.batchDistance(
        np.ascontiguousarray(src, dtype=np.float32),
        np.ascontiguousarray(dst, dtype=np.float32),
        cv2.CV_32F, normType=cv2.NORM_L2SQR, K=1)
    return indices[:, 0], distances[:, 0]


def icp(src: np.ndarray, dst: np.ndarray,
        max_dist: float, max_iter: int = 30, tol: float = 1e-5):
    """
    Align src → dst. Returns (T_corr, rmse).
    T_corr transforms the coordinates supplied in src into dst's frame.
    Infinite error marks insufficient overlap rather than a valid identity fit.
    """
    T_acc = np.eye(3)
    if len(src) < 6 or len(dst) < 6:
        return T_acc, math.inf
    pts   = src.copy()
    prev_err = 1e9

    for _ in range(max_iter):
        nn_idx, nn_d2 = _nearest_neighbors(pts, dst)

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

    _, final_distances = _nearest_neighbors(pts, dst)
    inlier = final_distances < max_dist ** 2
    # Six coincident points in otherwise unrelated scans must not turn into
    # a confident pose update and then contaminate subsequent keyframes.
    minimum_overlap = max(6, math.ceil(len(src) * 0.5))
    rmse = (float(np.sqrt(final_distances[inlier].mean()))
            if inlier.sum() >= minimum_overlap else math.inf)
    return T_acc, rmse


def surface_normals(points):
    """Estimate line normals; corners and poorly supported patches get zero."""
    if len(points) < 12:
        return np.zeros_like(points)
    cloud = np.ascontiguousarray(points, dtype=np.float32)
    _, neighbors = cv2.batchDistance(
        cloud, cloud, cv2.CV_32F, normType=cv2.NORM_L2SQR, K=5)
    patches = points[neighbors]
    patches = patches - patches.mean(axis=1, keepdims=True)
    covariances = np.einsum('nki,nkj->nij', patches, patches)
    values, vectors = np.linalg.eigh(covariances)
    lines = ((values[:, 1] > 1e-4)
             & (values[:, 0] < 0.2 * values[:, 1]))
    normals = vectors[:, :, 0]
    normals[~lines] = 0
    return normals


def observable_geometry(points, normals=None):
    """Require independent surface directions before trusting planar motion."""
    if normals is None:
        normals = surface_normals(points)
    normals = normals[np.linalg.norm(normals, axis=1) > 0.5]
    if len(normals) < 10:
        return False
    support = np.linalg.eigvalsh(normals.T @ normals)
    # A single wall or parallel corridor cannot constrain motion along it.
    return bool(support[0] > 0.02 * support[1])


def align_surfaces(src, target, normals, max_distance):
    """Align to wall surfaces without voxel spacing bias along each wall."""
    points = src.copy()
    accumulated = np.eye(3)
    solved = False
    for _ in range(12):
        indices, distances = _nearest_neighbors(points, target)
        selected_normals = normals[indices]
        valid = ((distances < max_distance ** 2)
                 & (np.linalg.norm(selected_normals, axis=1) > 0.5))
        if np.count_nonzero(valid) < 10:
            break
        source = points[valid]
        normal = selected_normals[valid]
        residual = np.sum(normal * (source - target[indices[valid]]), axis=1)
        pivot = source.mean(axis=0)
        relative = source - pivot
        jacobian = np.column_stack((normal,
            -normal[:, 0] * relative[:, 1]
            + normal[:, 1] * relative[:, 0]))
        weights = cauchy_weight(residual * residual, max_distance * 0.25)
        hessian = jacobian.T @ (weights[:, None] * jacobian)
        if np.linalg.cond(hessian) > 1e8:
            break
        try:
            step = np.linalg.solve(hessian, -jacobian.T @ (weights * residual))
        except np.linalg.LinAlgError:
            break
        if (not np.isfinite(step).all() or np.linalg.norm(step[:2]) > 0.5
                or abs(step[2]) > 0.3):
            break
        transform = make_T(0.0, 0.0, float(step[2]))
        transform[:2, 2] = step[:2] + pivot - transform[:2, :2] @ pivot
        points = apply_T(transform, points)
        accumulated = transform @ accumulated
        solved = True
        if np.linalg.norm(step) < 1e-5:
            break
    return accumulated if solved else None


def registration_error(points, target, max_distance):
    """Evaluate overlap separately from the residual of surviving matches."""
    _, distances = _nearest_neighbors(points, target)
    inliers = distances < max_distance * max_distance
    if np.count_nonzero(inliers) < max(6, math.ceil(0.65 * len(points))):
        return math.inf
    return float(np.sqrt(distances[inliers].mean()))


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
        self.declare_parameter('max_linear_speed', 1.0)
        self.declare_parameter('max_angular_speed', 2.0)
        self.declare_parameter('tracking_rmse', 0.12)

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
        self.max_linear_speed = p('max_linear_speed').value
        self.max_angular_speed = p('max_angular_speed').value
        self.tracking_rmse = p('tracking_rmse').value

        # State
        self.pose_T         = np.eye(3)
        self.prev_stamp     = None
        self.last_kf_T      = None          # pose at last keyframe
        self.odom_T         = np.eye(3)     # latest wheel odom pose
        self.prev_odom_T    = np.eye(3)     # wheel odom pose at last scan
        self.have_odom      = False
        self.odom_history   = deque(maxlen=200)
        self.pending_scan   = None
        self._stationary_reference = None
        self._stationary_geometry = None
        self._cached_local_map = None
        self._cached_normals = None
        self._map_geometry_ok = False

        # Rolling window of keyframe scans in their own sensor frames,
        # plus the world-frame pose at the time they were captured.
        # Each entry: (pts_sensor_frame, T_world_from_sensor)
        self.keyframes: deque = deque(maxlen=self.map_window)

        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            LaserScan, '/lidar', self._scan_cb,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.pub = self.create_publisher(Odometry, '/lidar_odom', 10)
        self.get_logger().info('LiDAR-Odom node started.')

    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            return
        stamp_ns = self._stamp_ns(msg.header.stamp)
        if self.odom_history and stamp_ns < self.odom_history[-1][0]:
            if self.odom_history[-1][0] - stamp_ns < 1_000_000_000:
                return
            # A reset simulation clock also invalidates the old local map.
            self.odom_history.clear()
            self.keyframes.clear()
            self.prev_stamp = None
            self.last_kf_T = None
            self.pending_scan = None
            self._stationary_reference = None
            self._stationary_geometry = None
            self._cached_local_map = None
        if self.odom_history and stamp_ns == self.odom_history[-1][0]:
            self.odom_history.pop()
        self.odom_history.append((stamp_ns, (x, y, yaw)))
        self.odom_T = make_T(x, y, yaw)
        self.have_odom = True
        # If the scan arrived just before its matching wheel sample, process
        # it now. At most one pending scan is retained to avoid building lag.
        if (self.pending_scan is not None
                and self._stamp_ns(self.pending_scan.header.stamp) <= stamp_ns):
            pending, self.pending_scan = self.pending_scan, None
            self._scan_cb(pending)

    @staticmethod
    def _stamp_ns(stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def _odom_at(self, stamp_ns):
        """Interpolate wheel pose at the scan time, never at callback time."""
        after = None
        for sample_ns, pose in reversed(self.odom_history):
            if sample_ns == stamp_ns:
                return make_T(*pose)
            if sample_ns < stamp_ns:
                if after is None:
                    return None
                after_ns, after_pose = after
                fraction = (stamp_ns - sample_ns) / (after_ns - sample_ns)
                x = pose[0] + fraction * (after_pose[0] - pose[0])
                y = pose[1] + fraction * (after_pose[1] - pose[1])
                yaw = pose[2] + fraction * math.remainder(
                    after_pose[2] - pose[2], math.tau)
                return make_T(x, y, yaw)
            after = sample_ns, pose
        return None

    def _build_local_map(self) -> np.ndarray:
        """
        Combine all keyframe scans into a single point cloud in world frame,
        then downsample. This is the ICP target.
        """
        if self._cached_local_map is not None:
            return self._cached_local_map
        all_pts = []
        for pts_sensor, T_kf in self.keyframes:
            all_pts.append(apply_T(T_kf, pts_sensor))
        if not all_pts:
            return np.empty((0, 2))
        combined = np.vstack(all_pts)
        self._cached_local_map = voxel_downsample(combined, self.vsize)
        self._cached_normals = surface_normals(self._cached_local_map)
        self._map_geometry_ok = observable_geometry(
            self._cached_local_map, self._cached_normals)
        return self._cached_local_map

    def _stationary_scene(self, msg):
        """Compare against an anchored scan, not merely the preceding frame."""
        geometry = (len(msg.ranges), msg.angle_min, msg.angle_increment)
        if (self._stationary_reference is None
                or geometry != self._stationary_geometry):
            return False
        ranges = np.asarray(msg.ranges)
        reference = self._stationary_reference
        valid = (np.isfinite(ranges) & np.isfinite(reference)
                 & (ranges >= msg.range_min) & (reference >= msg.range_min)
                 & (ranges < min(msg.range_max, self.max_range))
                 & (reference < min(msg.range_max, self.max_range)))
        if np.count_nonzero(valid) < max(30, self.min_pts):
            return False
        difference = np.abs(ranges[valid] - reference[valid])
        return bool(np.median(difference) < 0.02
                    and np.quantile(difference, 0.8) < 0.04)

    def _anchor_scene(self, msg):
        self._stationary_reference = np.asarray(msg.ranges).copy()
        self._stationary_geometry = (len(msg.ranges), msg.angle_min,
                                     msg.angle_increment)

    def _scan_cb(self, msg: LaserScan):
        if not self.have_odom:
            self.pending_scan = msg
            return
        stamp = msg.header.stamp
        dt = None
        if self.prev_stamp is not None:
            dt = ((stamp.sec - self.prev_stamp.sec)
                  + (stamp.nanosec - self.prev_stamp.nanosec) * 1e-9)
            if dt <= 0.0:
                return
        scan_odom = self._odom_at(self._stamp_ns(stamp))
        if scan_odom is None:
            if self._stamp_ns(stamp) > self.odom_history[-1][0]:
                self.pending_scan = msg
            return
        pts_raw = scan_to_points(msg, self.max_range)
        pts = voxel_downsample(pts_raw, self.vsize)
        if len(pts) < self.min_pts:
            self.get_logger().warning(
                f'Scan too small for localization: {len(pts)} points.',
                throttle_duration_sec=2.0)
            self._publish(stamp, 0.0, 0.0, 0.0, valid=False)
            return

        if not self.keyframes:
            # Seeding alone establishes an origin, not a measured velocity.
            self.pose_T = scan_odom.copy()
            self.prev_odom_T = scan_odom.copy()
            self.keyframes.append((pts, self.pose_T.copy()))
            self._cached_local_map = None
            self.last_kf_T = self.pose_T.copy()
            self.prev_stamp = stamp
            self._anchor_scene(msg)
            self._publish(stamp, 0.0, 0.0, 0.0, valid=False)
            return

        map_world = self._build_local_map()
        if len(map_world) < self.min_pts or not self._map_geometry_ok:
            self.get_logger().warning(
                'LiDAR geometry cannot constrain planar motion; holding pose.',
                throttle_duration_sec=2.0)
            self._publish(stamp, 0.0, 0.0, 0.0, valid=False)
            return

        previous_pose = self.pose_T.copy()
        stationary = self._stationary_scene(msg)
        previous_points = apply_T(previous_pose, pts)
        rmse = registration_error(previous_points, map_world,
                                  self.icp_max_dist)
        if stationary and rmse <= self.tracking_rmse:
            # A stable scene is direct evidence against spinning wheel odom.
            # Keeping its anchor also prevents tiny per-scan biases adding up.
            candidate = previous_pose
        else:
            odom_delta = inv_T(self.prev_odom_T) @ scan_odom
            seeds = [previous_pose, previous_pose @ odom_delta]
            fits = []
            for seed in seeds:
                correction = align_surfaces(
                    apply_T(seed, pts), map_world, self._cached_normals,
                    self.icp_max_dist)
                if correction is None:
                    continue
                pose = correction @ seed
                fit = registration_error(apply_T(pose, pts), map_world,
                                         self.icp_max_dist)
                motion = inv_T(previous_pose) @ pose
                dx, dy, angle = T_to_pose(motion)
                elapsed = max(dt or 0.0, 0.05)
                if (np.isfinite(pose).all() and math.isfinite(fit)
                        and fit <= self.tracking_rmse
                        and math.hypot(dx, dy) <= max(
                            0.15, 2 * self.max_linear_speed * elapsed)
                        and abs(angle) <= max(
                            0.2, 2 * self.max_angular_speed * elapsed)):
                    fits.append((fit, math.hypot(dx, dy) + abs(angle), pose))
            if not fits:
                self.get_logger().warning(
                    'LiDAR registration lost; holding pose instead of '
                    'integrating unchecked wheel motion.',
                    throttle_duration_sec=2.0)
                self._publish(stamp, 0.0, 0.0, 0.0, valid=False)
                return
            # Prefer the smaller motion only among genuinely equivalent fits.
            best_error = min(fit[0] for fit in fits)
            _, _, candidate = min(
                (fit for fit in fits if fit[0] <= best_error + 0.003),
                key=lambda fit: fit[1])
            rmse = registration_error(apply_T(candidate, pts), map_world,
                                      self.icp_max_dist)
            self._anchor_scene(msg)

        motion = inv_T(previous_pose) @ candidate
        dx, dy, angle = T_to_pose(motion)
        elapsed = max(dt or 0.0, 1e-6)
        # This twist is measured from registered scans, never copied from wheels.
        vx, vy, wz = dx / elapsed, dy / elapsed, angle / elapsed
        self.prev_stamp = stamp
        self.prev_odom_T = scan_odom.copy()
        self.pose_T = candidate
        dx_kf, dy_kf, angle_kf = T_to_pose(inv_T(self.last_kf_T) @ candidate)
        if (math.hypot(dx_kf, dy_kf) > self.kf_dist
                or abs(angle_kf) > self.kf_angle):
            self.keyframes.append((pts, candidate.copy()))
            self.last_kf_T = candidate.copy()
            self._cached_local_map = None
        self._publish(stamp, vx, vy, wz, rmse=rmse, valid=True)

    def _publish(self, stamp, vx, vy, wz, rmse=0.05, valid=True):
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
        scale = max(1.0, rmse / 0.05) if valid else 1e6
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
        tc[0]  = self.cov_vx * scale
        tc[7]  = self.cov_vx * scale
        tc[14] = LARGE
        tc[21] = LARGE
        tc[28] = LARGE
        tc[35] = self.cov_vyaw * scale
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
