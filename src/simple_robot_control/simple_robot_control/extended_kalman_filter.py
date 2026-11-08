#!/usr/bin/env python3
# Authors: Akash Mohapatra, Manas Chintawar

import math
import time
from collections import deque

from geometry_msgs.msg import TransformStamped

from nav_msgs.msg import Odometry

import numpy as np

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node

from sensor_msgs.msg import Imu

from std_msgs.msg import Bool

from tf2_ros import TransformBroadcaster


class ExtendedKalmanFilterNode(Node):

    def __init__(self):
        super().__init__('ekf_node')

        self.state = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
        self._identity = np.eye(3)
        self.P = self._identity.copy()

        self.declare_parameter('process_noise_vx', 0.1)
        self.declare_parameter('process_noise_omega', 0.5)
        self.declare_parameter('imu_heading_noise', 0.05)

        process_vx = self.get_parameter('process_noise_vx').value
        process_omega = self.get_parameter('process_noise_omega').value
        self.R_imu = self.get_parameter('imu_heading_noise').value

        self.M = np.array([
            [process_vx, 0.0],
            [0.0, process_omega]
        ])

        self.last_odom_time = None
        self._last_odom_stamp_ns = None
        self._last_lidar_stamp_ns = None
        self._pending_lidar = None
        # Wheel poses let delayed scan measurements advance to the current
        # filter time instead of pulling the robot back to an earlier pose.
        self._odom_history = deque(maxlen=200)
        self._prediction_pose = np.zeros(3)
        self._scan_velocity = np.zeros(3)
        self._reported_velocity = np.zeros(3)
        self._scan_velocity_variances = np.full(3, 1e6)
        self._last_good_lidar_ns = None
        self._last_good_lidar_wall = None
        self._lidar_quality_good = False
        self._last_wheel_velocity = (0.0, 0.0)
        self._slip_count = 0
        self._wheel_slip = False
        self.scan_timeout = self.declare_parameter(
            'scan_timeout', 0.6).value
        self.slip_confirmation_scans = self.declare_parameter(
            'slip_confirmation_scans', 4).value
        self.publish_tf = self.declare_parameter('publish_tf', False).value
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            Odometry, '/odom', self.prediction_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(
            Odometry, '/lidar_odom', self.lidar_odom_callback, 10)

        self.fused_odom_pub = self.create_publisher(
            Odometry, '/odom_fused', 10)
        self.tracking_pub = self.create_publisher(
            Bool, '/localization/tracking_valid', 1)
        self.slip_pub = self.create_publisher(
            Bool, '/localization/wheel_slip', 1)
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._tracking_timer = self.create_timer(
            0.1, self._publish_tracking_state, clock=self._steady_clock)

        self.get_logger().info(
            'EKF Node started (Wheel Odometry + IMU + LiDAR Odometry).')

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    def normalize_angle(self, angle):
        if -math.pi <= angle <= math.pi:
            return angle
        return math.remainder(angle, math.tau)

    @staticmethod
    def _stamp_ns(stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def _tracking_healthy(self):
        return bool(self._lidar_quality_good
                    and self._last_good_lidar_wall is not None
                    and time.monotonic() - self._last_good_lidar_wall
                    <= self.scan_timeout
                    and self._last_good_lidar_ns is not None
                    and self._last_odom_stamp_ns is not None
                    and 0 <= (self._last_odom_stamp_ns
                              - self._last_good_lidar_ns)
                    <= self.scan_timeout * 1e9)

    def _publish_tracking_state(self):
        self.tracking_pub.publish(Bool(data=self._tracking_healthy()))
        self.slip_pub.publish(Bool(data=bool(self._wheel_slip)))

    @staticmethod
    def _velocity_disagrees(wheel_v, wheel_w, scan_velocity):
        vx, vy, omega = scan_velocity
        # Slip is wheels reporting *more* motion than the scene supports.
        # Scan velocity can legitimately spike when sub-cell motion becomes
        # observable, or lag during acceleration; neither warrants a stop.
        linear = (abs(wheel_v) - abs(vx)
                  > max(0.12, 0.5 * abs(wheel_v))
                  or (wheel_v * vx < 0 and abs(wheel_v) > 0.12
                      and abs(vx) > 0.08)
                  or abs(vy) > 0.15)
        angular = (abs(wheel_w) - abs(omega)
                   > max(0.25, 0.5 * abs(wheel_w))
                   or (wheel_w * omega < 0 and abs(wheel_w) > 0.25
                       and abs(omega) > 0.15))
        return bool(linear or angular)

    def _prediction_velocity(self, wheel_v, wheel_w):
        if not self._tracking_healthy():
            return 0.0, 0.0, 0.0
        # Scan velocity is averaged over the preceding scan interval. Brief
        # disagreement during acceleration is normal; persistent disagreement
        # confirms slip before it replaces the current wheel prediction.
        if self._wheel_slip:
            return tuple(self._scan_velocity)
        return wheel_v, 0.0, wheel_w

    def _record_odometry(self, msg, velocity=(0.0, 0.0, 0.0), dt=0.0):
        # Keep the relative prediction motion, not drifting raw wheel poses.
        # Thus delay compensation cannot reintroduce a rejected wheel delta.
        vx, vy, omega = velocity
        yaw = self._prediction_pose[2]
        c, s = math.cos(yaw), math.sin(yaw)
        self._prediction_pose[0] += (c * vx - s * vy) * dt
        self._prediction_pose[1] += (s * vx + c * vy) * dt
        self._prediction_pose[2] = self.normalize_angle(yaw + omega * dt)
        self._odom_history.append((self._stamp_ns(msg.header.stamp),
                                   tuple(self._prediction_pose)))

    def _wheel_pose_at(self, stamp_ns):
        """Interpolate accepted motion within the retained time interval."""
        after = None
        for sample_ns, pose in reversed(self._odom_history):
            if sample_ns == stamp_ns:
                return pose
            if sample_ns < stamp_ns:
                if after is None:
                    return None
                after_ns, after_pose = after
                fraction = (stamp_ns - sample_ns) / (after_ns - sample_ns)
                return (pose[0] + fraction * (after_pose[0] - pose[0]),
                        pose[1] + fraction * (after_pose[1] - pose[1]),
                        self.normalize_angle(pose[2] + fraction *
                                             self.normalize_angle(
                                                 after_pose[2] - pose[2])))
            after = sample_ns, pose
        return None

    def _compensate_lidar_delay(self, z, variances, stamp_ns,
                                motion_override=None):
        """Advance a scan-time pose and noise to the current filter time."""
        if not self._odom_history:
            return None
        latest_ns, latest_pose = self._odom_history[-1]
        if latest_ns != self._last_odom_stamp_ns:
            return None
        if stamp_ns > latest_ns:
            return None
        scan_pose = self._wheel_pose_at(stamp_ns)
        if scan_pose is None:
            return None
        if stamp_ns == latest_ns:
            return z, variances

        # Express accepted translation in its scan-time body frame, then
        # rotate it into the LiDAR measurement's corrected heading.
        dx = latest_pose[0] - scan_pose[0]
        dy = latest_pose[1] - scan_pose[1]
        relative_yaw = self.normalize_angle(latest_pose[2] - scan_pose[2])
        rotation = z[2] - scan_pose[2]
        c, s = math.cos(rotation), math.sin(rotation)
        delta_x, delta_y = c * dx - s * dy, s * dx + c * dy
        delay = (latest_ns - stamp_ns) * 1e-9
        if motion_override is not None:
            # A newly detected slip invalidates even the recent prediction
            # history. Advance that measurement from its own scan motion.
            vx, vy, omega = motion_override
            angle = omega * delay
            if abs(omega) < 1e-6:
                body_x, body_y = vx * delay, vy * delay
            else:
                a, b = math.sin(angle) / omega, (1 - math.cos(angle)) / omega
                body_x, body_y = a * vx - b * vy, b * vx + a * vy
            c, s = math.cos(z[2]), math.sin(z[2])
            delta_x = c * body_x - s * body_y
            delta_y = s * body_x + c * body_y
            relative_yaw = angle
        advanced = z.copy()
        advanced[0] += delta_x
        advanced[1] += delta_y
        advanced[2] = self.normalize_angle(advanced[2] + relative_yaw)

        # Start-heading uncertainty moves the translated endpoint sideways.
        # Apply the same conservative full-motion lever arm to process noise
        # accumulated over the delay; isotropic translation avoids assuming
        # that curved motion stayed on the original forward axis.
        jacobian = self._identity.copy()
        jacobian[0, 2] = -delta_y
        jacobian[1, 2] = delta_x
        noise_v = max(0.0, float(self.M[0, 0])) * delay * delay
        noise_yaw = max(0.0, float(self.M[1, 1])) * delay * delay
        delayed_variances = variances + (noise_v, noise_v, noise_yaw)
        covariance = (jacobian * delayed_variances) @ jacobian.T
        return advanced, covariance

    def _drain_pending_lidar(self):
        """Apply a queued scan once wheel history reaches its timestamp."""
        pending = self._pending_lidar
        if pending is None or self._last_odom_stamp_ns is None:
            return False
        if self._stamp_ns(pending.header.stamp) > self._last_odom_stamp_ns:
            return False
        self._pending_lidar = None
        previous_update = self._last_lidar_stamp_ns
        self.lidar_odom_callback(pending)
        return self._last_lidar_stamp_ns != previous_update

    def prediction_callback(self, msg):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if (self.last_odom_time is not None
                and current_time < self.last_odom_time - 1.0):
            # A simulation clock reset invalidates both the old state time
            # and every pose that could otherwise compensate a delayed scan.
            self.last_odom_time = None
            self._last_odom_stamp_ns = None
            self._last_lidar_stamp_ns = None
            self._pending_lidar = None
            self._odom_history.clear()
            self.P = self._identity.copy()
            self._prediction_pose.fill(0.0)
            self._scan_velocity.fill(0.0)
            self._reported_velocity.fill(0.0)
            self._last_good_lidar_ns = None
            self._last_good_lidar_wall = None
            self._lidar_quality_good = False
            self._slip_count = 0
            self._wheel_slip = False

        wheel_v = msg.twist.twist.linear.x
        wheel_w = msg.twist.twist.angular.z
        if not all(math.isfinite(value) for value in (wheel_v, wheel_w)):
            return
        self._last_wheel_velocity = (wheel_v, wheel_w)

        if self.last_odom_time is None:
            self.last_odom_time = current_time
            self._last_odom_stamp_ns = self._stamp_ns(msg.header.stamp)
            self.state[0] = msg.pose.pose.position.x
            self.state[1] = msg.pose.pose.position.y
            self.state[2] = self.quaternion_to_yaw(
                msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
            )
            self._record_odometry(msg)
            if self._drain_pending_lidar():
                self.publish_fused_odometry(msg.header.stamp)
            return

        dt = current_time - self.last_odom_time
        if dt <= 0.0:
            return
        self.last_odom_time = current_time
        self._last_odom_stamp_ns = self._stamp_ns(msg.header.stamp)
        vx, vy, omega = self._prediction_velocity(wheel_v, wheel_w)
        self._reported_velocity[:] = vx, vy, omega
        self._record_odometry(msg, (vx, vy, omega), dt)
        yaw_k = self.state[2]
        cos_dt = math.cos(yaw_k) * dt
        sin_dt = math.sin(yaw_k) * dt

        self.state[0] += vx * cos_dt - vy * sin_dt
        self.state[1] += vx * sin_dt + vy * cos_dt
        self.state[2] = self.normalize_angle(self.state[2] + omega * dt)

        # F is identity except for its yaw column, and M is diagonal.
        # Expand F P F.T + V M V.T to avoid allocating Jacobians and
        # multiplying four tiny matrices on every wheel-odometry callback.
        a = -vx * sin_dt - vy * cos_dt
        b = vx * cos_dt - vy * sin_dt
        p00, p01, p02 = self.P[0]
        _, p11, p12 = self.P[1]
        p22 = self.P[2, 2]
        linear_noise = self.M[0, 0]
        lateral_noise = linear_noise if vy else 0.0
        self.P[0, 0] = (
            p00 + 2.0 * a * p02 + a * a * p22
            + linear_noise * cos_dt * cos_dt + lateral_noise * sin_dt * sin_dt)
        self.P[0, 1] = self.P[1, 0] = (
            p01 + b * p02 + a * p12 + a * b * p22
            + (linear_noise - lateral_noise) * cos_dt * sin_dt)
        self.P[0, 2] = self.P[2, 0] = p02 + a * p22
        self.P[1, 1] = (
            p11 + 2.0 * b * p12 + b * b * p22
            + linear_noise * sin_dt * sin_dt + lateral_noise * cos_dt * cos_dt)
        self.P[1, 2] = self.P[2, 1] = p12 + b * p22
        self.P[2, 2] = p22 + self.M[1, 1] * dt * dt

        self._drain_pending_lidar()
        self.publish_fused_odometry(msg.header.stamp)

    def imu_callback(self, msg):
        # sensor_msgs/Imu uses -1 to mark an unavailable orientation.
        if msg.orientation_covariance[0] < 0.0:
            return
        imu_yaw = self.quaternion_to_yaw(
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w
        )

        y_val = self.normalize_angle(imu_yaw - self.state[2])

        R = (msg.orientation_covariance[8]
             if msg.orientation_covariance[8] > 0.0 else self.R_imu)

        # H = [0, 0, 1]: the innovation variance is scalar and the gain
        # only needs the yaw column of P.
        S = self.P[2, 2] + R
        if not math.isfinite(S) or S <= 0.0:
            return
        K = self.P[:, 2] / S

        self.state += K * y_val
        self.state[2] = self.normalize_angle(self.state[2])
        IKH = self._identity.copy()
        IKH[:, 2] -= K
        # Keep the Joseph form rather than a subtractive covariance update.
        self.P = IKH @ self.P @ IKH.T + R * np.outer(K, K)
        self.P = (self.P + self.P.T) * 0.5

    def lidar_odom_callback(self, msg):
        """UPDATE STEP: Fuse LiDAR odometry pose (x, y, yaw) into EKF."""
        stamp_ns = self._stamp_ns(msg.header.stamp)
        if (self._last_lidar_stamp_ns is not None
                and stamp_ns <= self._last_lidar_stamp_ns):
            return
        lidar_yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        )

        z = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            lidar_yaw
        ])
        if not np.isfinite(z).all() or not np.isfinite(self.state).all():
            self._lidar_quality_good = False
            self._publish_tracking_state()
            return

        # Read covariance from lidar_odom message (set in lidar_odom.py)
        variances = np.array([
            msg.pose.covariance[0],   # x noise
            msg.pose.covariance[7],   # y noise
            msg.pose.covariance[35]   # yaw noise
        ])
        scan_velocity = np.array([msg.twist.twist.linear.x,
                                  msg.twist.twist.linear.y,
                                  msg.twist.twist.angular.z])
        velocity_variances = np.array([msg.twist.covariance[0],
                                       msg.twist.covariance[7],
                                       msg.twist.covariance[35]])
        if (not np.isfinite(variances).all() or np.any(variances <= 0.0)
                or np.any(variances > 0.25)
                or not np.isfinite(scan_velocity).all()
                or not np.isfinite(velocity_variances).all()
                or np.any(velocity_variances <= 0.0)
                or np.any(velocity_variances > 0.5)):
            self._lidar_quality_good = False
            self._publish_tracking_state()
            return
        if (self._last_odom_stamp_ns is None
                or stamp_ns > self._last_odom_stamp_ns):
            if (self._pending_lidar is None
                    or stamp_ns > self._stamp_ns(
                        self._pending_lidar.header.stamp)):
                self._pending_lidar = msg
            return
        if (self._last_odom_stamp_ns - stamp_ns) * 1e-9 > self.scan_timeout:
            self._lidar_quality_good = False
            self._publish_tracking_state()
            return
        disagreement = self._velocity_disagrees(
            *self._last_wheel_velocity, scan_velocity)
        motion_override = (scan_velocity if disagreement
                           or not self._lidar_quality_good else None)
        compensated = self._compensate_lidar_delay(
            z, variances, stamp_ns, motion_override)
        if compensated is None:
            self._lidar_quality_good = False
            self._publish_tracking_state()
            return
        z, measurement_noise = compensated
        innovation = z - self.state
        innovation[2] = self.normalize_angle(innovation[2])
        was_slipping = self._wheel_slip
        self._slip_count = self._slip_count + 1 if disagreement else 0
        self._wheel_slip = self._slip_count >= self.slip_confirmation_scans
        if self._wheel_slip:
            # A drifting wheel prior must not overpower independent scan pose.
            self.P[:2, :2] += np.eye(2) * max(variances[:2]) * 10
            if not was_slipping:
                self.get_logger().warning(
                    'Wheel slip detected; using registered LiDAR motion.')

        # H is identity, so solve K S = P directly. Avoid explicitly
        # inverting S and avoid constructing a diagonal R twice.
        S = self.P.copy()
        if measurement_noise.ndim == 1:
            S.flat[::4] += measurement_noise
        else:
            S += measurement_noise
        if not np.isfinite(S).all():
            return
        try:
            K = np.linalg.solve(S.T, self.P.T).T
        except np.linalg.LinAlgError:
            # A degenerate zero-noise observation must not stop the node.
            return

        self.state += K @ innovation
        self.state[2] = self.normalize_angle(self.state[2])
        IKH = self._identity - K
        if measurement_noise.ndim == 1:
            correction_noise = (K * measurement_noise) @ K.T
        else:
            correction_noise = K @ measurement_noise @ K.T
        self.P = IKH @ self.P @ IKH.T + correction_noise
        self.P = (self.P + self.P.T) * 0.5
        if (self._wheel_slip and np.linalg.norm(scan_velocity[:2]) < 0.03
                and abs(scan_velocity[2]) < 0.06):
            # Remove the residual wheel offset while pressed against a wall.
            self.state[:] = z
        self._last_lidar_stamp_ns = stamp_ns
        self._scan_velocity = scan_velocity
        self._scan_velocity_variances = velocity_variances
        self._last_good_lidar_ns = stamp_ns
        self._last_good_lidar_wall = time.monotonic()
        self._lidar_quality_good = True
        self._reported_velocity[:] = self._prediction_velocity(
            *self._last_wheel_velocity)
        self._publish_tracking_state()

    def publish_fused_odometry(self, timestamp):
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'

        odom.pose.pose.position.x = float(self.state[0])
        odom.pose.pose.position.y = float(self.state[1])
        odom.pose.pose.position.z = 0.0

        qx, qy, qz, qw = self.yaw_to_quaternion(self.state[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.pose.covariance[0] = self.P[0, 0]
        odom.pose.covariance[7] = self.P[1, 1]
        odom.pose.covariance[35] = self.P[2, 2]

        vx, vy, omega = self._reported_velocity
        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        odom.twist.twist.angular.z = float(omega)
        for index, variance in zip((0, 7, 35), self._scan_velocity_variances):
            odom.twist.covariance[index] = float(variance)

        self.fused_odom_pub.publish(odom)
        self._publish_tracking_state()
        if self.publish_tf:
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = odom.child_frame_id
            transform.transform.translation.x = float(self.state[0])
            transform.transform.translation.y = float(self.state[1])
            transform.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = ExtendedKalmanFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
