#!/usr/bin/env python3
# Authors: Akash Mohapatra, Manas Chintawar

import math
import signal
import time
from collections import deque

import cv2

from geometry_msgs.msg import Twist, Vector3

from nav_msgs.msg import OccupancyGrid, Odometry, Path

import numpy as np

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time

from sensor_msgs.msg import LaserScan

from simple_robot_control.path_planner import PathPlanner

from std_msgs.msg import Bool, Empty

from tf2_ros import Buffer, TransformException, TransformListener


def euler_from_quaternion(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


class PurePursuit(Node):
    """Track the global route using collision-checked local trajectories."""

    def __init__(self):
        super().__init__('pure_pursuit')

        def parameter(name, value):
            return self.declare_parameter(name, value).value

        self.robot_length = parameter('robot_length', 0.46)
        self.robot_width = parameter('robot_width', 0.34)
        self.safety_margin = parameter('safety_margin', 0.06)
        self.drive_axle_offset = parameter('drive_axle_offset_x', 0.10)
        self.MAX_DRIVE_SPEED = parameter('max_speed', 0.75)
        self.MAX_TURN_SPEED = parameter('max_turn_speed', 1.0)
        self.LOOKAHEAD_DISTANCE = parameter('lookahead_distance', 1.0)
        self.DISTANCE_TOLERANCE = parameter('goal_tolerance', 0.25)
        self.max_acceleration = parameter('max_acceleration', 0.9)
        self.braking_deceleration = parameter('braking_deceleration', 0.8)
        self.angular_acceleration = parameter('angular_acceleration', 2.0)
        self.prediction_horizon = parameter('prediction_horizon', 1.2)
        self.scan_timeout = parameter('scan_timeout', 0.5)
        self.odom_timeout = parameter('odom_timeout', 0.35)
        self.localization_timeout = parameter('localization_timeout', 0.75)
        self.navigator_timeout = parameter('navigator_timeout', 1.5)
        self.reaction_time = parameter('reaction_time', 0.2)
        self.blocked_delay = parameter('blocked_delay', 0.35)
        self.stall_timeout = parameter('stall_timeout', 2.0)
        self.base_frame = parameter('base_frame', 'base_footprint')
        positive = (self.robot_length, self.robot_width, self.MAX_DRIVE_SPEED,
                    self.MAX_TURN_SPEED, self.max_acceleration,
                    self.braking_deceleration, self.angular_acceleration,
                    self.prediction_horizon, self.scan_timeout,
                    self.odom_timeout, self.localization_timeout,
                    self.navigator_timeout, self.stall_timeout,
                    self.LOOKAHEAD_DISTANCE, self.DISTANCE_TOLERANCE,
                    self.blocked_delay)
        if (not all(math.isfinite(x) and x > 0 for x in positive)
                or not math.isfinite(self.drive_axle_offset)
                or not math.isfinite(self.safety_margin)
                or self.safety_margin < 0 or self.reaction_time < 0
                or not math.isfinite(self.reaction_time)):
            raise ValueError(
                'Footprint, motion limits and timeouts are invalid')

        self.pose = None
        self.mapdata = None
        self.path = Path()
        self._frame_id = None
        self._path_frame = None
        self._map_frame = None
        self._path_valid = False
        self._map_grid = None
        self._map_occupied_xy = np.empty((0, 2))
        self._path_xy = np.empty((0, 2))
        self._pose_history = deque(maxlen=100)
        self._scan_points_world = np.empty((0, 2))
        self._scan_geometry = None
        self._mount_frame = None
        self._mount_pose = None
        self._last_scan_ns = None
        self._pending_scan = None
        self._scan_valid = False
        self._localization_valid = False
        self._wheel_slip = True
        self._last_scan_time = -math.inf
        self._last_odom_time = -math.inf
        self._last_localization_time = -math.inf
        self._last_heartbeat_time = -math.inf
        self._last_step_time = None
        self._command_speed = 0.0
        self._command_turn = 0.0
        self._observed_speed = 0.0
        self._observed_forward_speed = 0.0
        self._observed_turn = 0.0
        self._start_recovery_requested = False
        self._start_recovery_received = -math.inf
        self._reversing = False
        self._reverse_start_time = 0.0
        self._reverse_duration = 1.2
        self._reverse_speed = -0.25
        self._recovery_needed = False
        self._recovery_status_time = -math.inf
        self._recovery_failed_notified = False
        self._blocked_since = None
        self._blocked = False
        self._last_status_time = -math.inf
        self._last_block_warning_time = -math.inf
        self._progress_anchor = None
        self._progress_time = self._now()
        self._stall_latched = False
        self._control_period = 0.05
        self._rollout_dt = 0.1
        self.min_x = -self.robot_length / 2
        self.max_x = self.robot_length / 2
        self.min_y = -self.robot_width / 2
        self.max_y = self.robot_width / 2
        self._tf_geometry_initialized = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.blocked_pub = self.create_publisher(
            Bool, '/navigation/blocked', 10)
        self.recovery_needed_pub = self.create_publisher(
            Bool, '/navigation/recovery_needed', 1)
        self.recovery_failed_pub = self.create_publisher(
            Bool, '/navigation/recovery_failed', 1)
        self.create_subscription(
            Odometry, '/odom_fused', self.update_odometry, 10)
        self.create_subscription(OccupancyGrid, '/map', self.update_map, 10)
        self.create_subscription(
            Path, '/pure_pursuit/path', self.update_path, 10)
        self.create_subscription(
            Bool, '/localization/tracking_valid', self.update_localization, 10)
        self.create_subscription(
            Bool, '/localization/wheel_slip', self.update_wheel_slip, 10)
        self.create_subscription(
            Empty, '/navigation/heartbeat', self.update_heartbeat, 10)
        self.create_subscription(
            Bool, '/navigation/recover_start', self.update_start_recovery, 1)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            LaserScan, '/lidar_2d_scan', self.update_lidar, sensor_qos)
        self._safety_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(
            self._control_period, self.run_step, clock=self._safety_clock)
        self.get_logger().info(
            'Runtime avoidance started with footprint and braking checks.')

    @staticmethod
    def _now():
        return time.monotonic()

    @staticmethod
    def _stamp_ns(stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def update_localization(self, msg):
        self._localization_valid = bool(msg.data)
        self._last_localization_time = self._now()

    def update_wheel_slip(self, msg):
        self._wheel_slip = bool(msg.data)

    def update_heartbeat(self, _msg):
        self._last_heartbeat_time = self._now()

    def _start_reverse(self, now, reason='blocked'):
        if not self._reversing:
            self._reversing = True
            self._reverse_start_time = now
            self.get_logger().info(f'Stuck or blocked ({reason}); reversing to recover clearance.')

    def update_start_recovery(self, msg):
        """Navigator requested clearance recovery; execute reverse maneuver if needed."""
        now = self._now()
        requested = bool(msg.data)
        self._start_recovery_requested = requested
        self._start_recovery_received = now
        if requested and not self._reversing:
            self._start_reverse(now, reason='navigator recovery request')

    def update_odometry(self, msg):
        pose = msg.pose.pose
        coordinates = (pose.position.x, pose.position.y,
                       euler_from_quaternion(pose.orientation))
        if (not msg.header.frame_id
                or not all(math.isfinite(x) for x in coordinates)):
            self.pose = None
            return
        stamp_ns = self._stamp_ns(msg.header.stamp)
        if not self._pose_history or stamp_ns >= self._pose_history[-1][0]:
            # Pose corrections are not physical acceleration; retain the
            # fused scan-checked twist even when a stop command is issued.
            velocity = msg.twist.twist.linear
            measured_speed = math.hypot(velocity.x, velocity.y)
            if math.isfinite(measured_speed):
                self._observed_speed = measured_speed
                self._observed_forward_speed = float(velocity.x)
            observed_turn = msg.twist.twist.angular.z
            if math.isfinite(observed_turn):
                self._observed_turn = float(observed_turn)
        if self._pose_history:
            previous_ns, previous = self._pose_history[-1]
            if stamp_ns < previous_ns:
                if previous_ns - stamp_ns < 1_000_000_000:
                    return
                self._pose_history.clear()
                self._scan_valid = self._path_valid = False
                self._last_scan_ns = None
                self._pending_scan = None
            elif stamp_ns > previous_ns:
                displacement = math.hypot(coordinates[0] - previous[0],
                                          coordinates[1] - previous[1])
                turn = abs(math.remainder(
                    coordinates[2] - previous[2], math.tau))
                if displacement > 0.4 or turn > 0.5:
                    self._scan_valid = False
            else:
                self._pose_history.pop()
        if (self._frame_id is not None
                and msg.header.frame_id != self._frame_id):
            self._scan_valid = self._path_valid = False
            self._pose_history.clear()
            self._pending_scan = None
        self._frame_id = msg.header.frame_id
        self._pose_history.append((stamp_ns, coordinates))
        self.pose = pose
        self._last_odom_time = self._now()
        if (self._pending_scan is not None and self._stamp_ns(
                self._pending_scan.header.stamp) <= stamp_ns):
            pending, self._pending_scan = self._pending_scan, None
            self.update_lidar(pending)

    def update_map(self, msg):
        size = msg.info.width * msg.info.height
        if (not size or size != len(msg.data) or msg.info.resolution <= 0
                or not msg.header.frame_id):
            self.mapdata = None
            return
        origin = msg.info.origin
        if (not all(math.isfinite(x) for x in (
                origin.position.x, origin.position.y, msg.info.resolution))
                or abs(origin.orientation.x) > 1e-6
                or abs(origin.orientation.y) > 1e-6
                or abs(origin.orientation.z) > 1e-6):
            self.mapdata = None
            return
        self._map_grid = np.asarray(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        self._recovery_cspace = None
        self._recovery_distance = None
        ys, xs = np.nonzero(self._map_grid >= 50)
        self._map_occupied_xy = np.column_stack((
            origin.position.x + (xs + 0.5) * msg.info.resolution,
            origin.position.y + (ys + 0.5) * msg.info.resolution))
        self.mapdata = msg
        self._map_frame = msg.header.frame_id

    def update_path(self, msg):
        self.path = msg
        self._path_xy = np.asarray([
            (p.pose.position.x, p.pose.position.y) for p in msg.poses
        ], dtype=np.float64).reshape(-1, 2)
        self._path_frame = msg.header.frame_id
        self._path_valid = (bool(msg.poses) and bool(msg.header.frame_id)
                            and np.isfinite(self._path_xy).all())
        self._progress_anchor = None
        self._progress_time = self._now()
        self._stall_latched = False
        if not msg.poses and not self._reversing:
            self.stop()
            self._clear_blocked(self._now())

    def _update_body_geometry_from_tf(self):
        """Update robot body bounds and drive axle dynamically from the TF2 tree."""
        if self._tf_geometry_initialized:
            return
        links = [
            'base_link', 'left_wheel', 'right_wheel', 'caster_wheel',
            'camera_left_link', 'camera_right_link', 'lidar_link'
        ]
        extents = {
            'base_link': (0.20, 0.15),
            'left_wheel': (0.05, 0.02),
            'right_wheel': (0.05, 0.02),
            'caster_wheel': (0.025, 0.025),
            'camera_left_link': (0.015, 0.04),
            'camera_right_link': (0.015, 0.04),
            'lidar_link': (0.05, 0.05),
        }
        xs, ys = [], []
        axle_x = None
        for link in links:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.base_frame, link, Time()).transform
                tx, ty = trans.translation.x, trans.translation.y
                hx, hy = extents.get(link, (0.02, 0.02))
                xs.extend([tx - hx, tx + hx])
                ys.extend([ty - hy, ty + hy])
                if 'wheel' in link and axle_x is None:
                    axle_x = tx
            except Exception:
                continue
        if xs and ys:
            self.min_x = min(xs)
            self.max_x = max(xs)
            self.min_y = min(ys)
            self.max_y = max(ys)
            self.robot_length = self.max_x - self.min_x
            self.robot_width = self.max_y - self.min_y
            if axle_x is not None:
                self.drive_axle_offset = axle_x
            self._tf_geometry_initialized = True
            self.get_logger().info(
                f'PurePursuit TF2 body model: length={self.robot_length:.2f}m, '
                f'width={self.robot_width:.2f}m, axle_offset={self.drive_axle_offset:.2f}m, '
                f'bounds=({self.min_x:.2f}, {self.max_x:.2f}, {self.min_y:.2f}, {self.max_y:.2f})')

    def _sensor_mount(self, frame):
        if frame == self.base_frame:
            return 0.0, 0.0, 0.0
        if not frame:
            return None
        if frame != self._mount_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame, frame, Time()).transform
            except TransformException:
                return None
            self._mount_pose = (transform.translation.x,
                                transform.translation.y,
                                euler_from_quaternion(transform.rotation))
            self._mount_frame = frame
        return self._mount_pose

    def _scan_pose_at(self, stamp_ns):
        after = None
        for sample_ns, pose in reversed(self._pose_history):
            if sample_ns == stamp_ns:
                return pose
            if sample_ns < stamp_ns:
                if after is None:
                    return None
                after_ns, next_pose = after
                fraction = (stamp_ns - sample_ns) / (after_ns - sample_ns)
                return (pose[0] + fraction * (next_pose[0] - pose[0]),
                        pose[1] + fraction * (next_pose[1] - pose[1]),
                        pose[2] + fraction * math.remainder(
                            next_pose[2] - pose[2], math.tau))
            after = sample_ns, pose
        return None

    def update_lidar(self, msg):
        stamp_ns = self._stamp_ns(msg.header.stamp)
        if self._last_scan_ns is not None and stamp_ns <= self._last_scan_ns:
            return
        valid_geometry = (len(msg.ranges) > 2
                          and math.isfinite(msg.angle_min)
                          and math.isfinite(msg.angle_increment)
                          and msg.angle_increment > 0
                          and math.isfinite(msg.range_max)
                          and msg.range_max > 0)
        coverage = (len(msg.ranges) - 1) * msg.angle_increment
        valid_geometry = (valid_geometry
                          and abs(coverage - math.tau) < 0.1)
        if not valid_geometry:
            self._scan_valid = False
            return
        ranges = np.asarray(msg.ranges)
        valid = (np.isfinite(ranges) & (ranges >= max(0.0, msg.range_min))
                 & (ranges > 0) & (ranges < msg.range_max))
        # A projected cloud with no retained points is unobserved space.
        if np.count_nonzero(valid) < 3:
            self._scan_valid = False
            return
        if not self._pose_history or stamp_ns > self._pose_history[-1][0]:
            if (self._pending_scan is None or stamp_ns >= self._stamp_ns(
                    self._pending_scan.header.stamp)):
                self._pending_scan = msg
            return
        pose = self._scan_pose_at(stamp_ns)
        mount = self._sensor_mount(msg.header.frame_id)
        if pose is None or mount is None:
            self._scan_valid = False
            return
        geometry = (len(msg.ranges), msg.angle_min, msg.angle_increment)
        if geometry != self._scan_geometry:
            angles = (msg.angle_min
                      + np.arange(len(msg.ranges)) * msg.angle_increment)
            self._beam_cos, self._beam_sin = np.cos(angles), np.sin(angles)
            self._scan_geometry = geometry
        px = ranges[valid] * self._beam_cos[valid]
        py = ranges[valid] * self._beam_sin[valid]
        mx, my, mount_yaw = mount
        c, s = math.cos(mount_yaw), math.sin(mount_yaw)
        bx, by = mx + c * px - s * py, my + s * px + c * py
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        self._scan_points_world = np.column_stack((
            x + c * bx - s * by, y + s * bx + c * by))
        self._scan_valid = np.isfinite(self._scan_points_world).all()
        self._last_scan_ns = stamp_ns
        self._last_scan_time = self._now()

    def send_speed(self, linear, angular):
        self._command_speed = float(linear)
        self._command_turn = float(angular)
        self.cmd_vel.publish(Twist(
            linear=Vector3(x=float(linear)),
            angular=Vector3(z=float(angular))))

    def stop(self):
        self.send_speed(0.0, 0.0)

    def _publish_blocked(self, blocked, now):
        if blocked != self._blocked or now - self._last_status_time > 0.25:
            self.blocked_pub.publish(Bool(data=bool(blocked)))
            self._last_status_time = now
        self._blocked = blocked

    def _note_blocked(self, now, immediate=False, reason=None):
        if self._blocked_since is None:
            self._blocked_since = now
        blocked = immediate or now - self._blocked_since >= self.blocked_delay
        self._publish_blocked(blocked, now)
        if blocked and reason and now - self._last_block_warning_time >= 3.0:
            self.get_logger().warning(reason)
            self._last_block_warning_time = now

    def _clear_blocked(self, now):
        self._blocked_since = None
        self._publish_blocked(False, now)

    def _publish_recovery_needed(self, needed, now):
        if needed != self._recovery_needed or now - self._recovery_status_time > 0.25:
            self.recovery_needed_pub.publish(Bool(data=bool(needed)))
            self._recovery_status_time = now
        self._recovery_needed = needed

    def _local_obstacles(self, world_pose, radius):
        points = []
        for source in (self._scan_points_world, self._map_occupied_xy):
            if len(source):
                points.append(source[np.sum(
                    (source - world_pose[:2]) ** 2, axis=1) < radius ** 2])
        return np.concatenate(points) if points else np.empty((0, 2))

    def _report_local_blockage(self, world_pose, now):
        # A blocked turn near the body needs room to manoeuvre, rather than
        # another frontier target from the same pose.
        obstacles = self._local_obstacles(world_pose, 1.5)
        reserve = (math.hypot(self.robot_length / 2 + abs(self.drive_axle_offset),
                              self.robot_width / 2)
                   + self.safety_margin + self.mapdata.info.resolution / math.sqrt(2) + 0.02)
        axle = np.array([world_pose[0] + self.drive_axle_offset * math.cos(world_pose[2]),
                         world_pose[1] + self.drive_axle_offset * math.sin(world_pose[2])])
        nearby = len(obstacles) and np.min(np.sum(
            (obstacles - axle) ** 2, axis=1)) < reserve ** 2
        self._publish_recovery_needed(bool(nearby), now)

    def _target(self, x, y):
        distances = np.sum((self._path_xy - (x, y)) ** 2, axis=1)
        nearest = int(np.argmin(distances))
        lookahead = self.LOOKAHEAD_DISTANCE
        further = np.flatnonzero(distances[nearest:] > lookahead ** 2)
        target = (nearest + int(further[0]) if len(further)
                  else len(distances) - 1)
        return self._path_xy[target], self._path_xy[nearest:]

    def _rollouts(self, speeds, turns, initial_speed, initial_turn,
                  horizon=0.6, step_dt=0.05):
        """Simulate short forward rollouts at normal speed without braking tails."""
        dt = step_dt
        horizon_steps = math.ceil(horizon / dt)
        trajectories = np.zeros((len(speeds), horizon_steps + 1, 3))
        v = np.full(len(speeds), initial_speed)
        w = np.full(len(speeds), initial_turn)
        for step in range(1, trajectories.shape[1]):
            target_v, target_w = speeds, turns
            accelerating = ((np.abs(target_v) > np.abs(v))
                            & (np.asarray(target_v) * v >= 0))
            rate = np.where(accelerating, self.max_acceleration,
                            self.braking_deceleration)
            v_next = v + np.clip(target_v - v, -rate * dt, rate * dt)
            w_next = w + np.clip(target_w - w,
                                 -self.angular_acceleration * dt,
                                 self.angular_acceleration * dt)
            mean_v, mean_w = (v + v_next) * 0.5, (w + w_next) * 0.5
            heading = trajectories[:, step - 1, 2] + mean_w * dt * 0.5
            trajectories[:, step, 0] = (
                trajectories[:, step - 1, 0] + mean_v * np.cos(heading) * dt)
            trajectories[:, step, 1] = (
                trajectories[:, step - 1, 1] + mean_v * np.sin(heading) * dt)
            trajectories[:, step, 2] = (
                trajectories[:, step - 1, 2] + mean_w * dt)
            previous_heading = trajectories[:, step - 1, 2]
            next_heading = trajectories[:, step, 2]
            trajectories[:, step, 0] += self.drive_axle_offset * (
                np.cos(previous_heading) - np.cos(next_heading))
            trajectories[:, step, 1] += self.drive_axle_offset * (
                np.sin(previous_heading) - np.sin(next_heading))
            v, w = v_next, w_next
        return trajectories, horizon_steps

    def _collision_scores(self, trajectories, world_pose, obstacles, margin):
        """Check footprint clearance against map cells and live obstacles."""
        x, y, yaw = world_pose
        c, s = math.cos(yaw), math.sin(yaw)
        relative_x = trajectories[:, :, 0]
        relative_y = trajectories[:, :, 1]
        wx = x + c * relative_x - s * relative_y
        wy = y + s * relative_x + c * relative_y
        heading = trajectories[:, :, 2] + yaw
        info = self.mapdata.info
        gx = np.floor((wx - info.origin.position.x)
                      / info.resolution).astype(np.int32)
        gy = np.floor((wy - info.origin.position.y)
                      / info.resolution).astype(np.int32)
        in_bounds = ((gx >= 0) & (gx < info.width)
                     & (gy >= 0) & (gy < info.height))
        values = self._map_grid[np.clip(gy, 0, info.height - 1),
                                np.clip(gx, 0, info.width - 1)]
        known = in_bounds & (values >= 0) & (values < 50)
        safe = np.all(known, axis=1)
        minimum_clearance = np.full(len(trajectories), 2.0)
        if not len(obstacles):
            return safe, minimum_clearance
        reserves = np.broadcast_to(margin, (len(trajectories),))
        for start in range(0, len(trajectories), 16):
            end = start + 16
            dx = obstacles[None, None, :, 0] - wx[start:end, :, None]
            dy = obstacles[None, None, :, 1] - wy[start:end, :, None]
            ch = np.cos(heading[start:end, :, None])
            sh = np.sin(heading[start:end, :, None])
            bx = ch * dx + sh * dy
            by = -sh * dx + ch * dy
            longitudinal = np.maximum(np.maximum(self.min_x - bx, bx - self.max_x), 0)
            lateral = np.maximum(np.maximum(self.min_y - by, by - self.max_y), 0)
            squared_clearance = longitudinal * longitudinal + lateral * lateral
            minimum_squared = np.min(squared_clearance, axis=(1, 2))
            safe[start:end] &= minimum_squared > reserves[start:end] ** 2
            minimum_clearance[start:end] = np.sqrt(minimum_squared)
        return safe, minimum_clearance

    def _choose_command(self, target, path, world_pose, scan_age, dt):
        """Choose normal drive speed and turn command without artificial braking slowdowns."""
        x, y, yaw = world_pose
        dx, dy = target[0] - x, target[1] - y
        alpha = math.remainder(math.atan2(dy, dx) - yaw, math.tau)
        distance = math.hypot(dx, dy)
        goal_distance = math.hypot(path[-1, 0] - x, path[-1, 1] - y)

        if goal_distance < self.DISTANCE_TOLERANCE:
            return 0.0, 0.0, True
        elif goal_distance < 0.4:
            cruise_speed = max(0.25, (goal_distance / 0.4) * self.MAX_DRIVE_SPEED)
        else:
            cruise_speed = self.MAX_DRIVE_SPEED

        # Large heading error: rotate in place to align with path
        if abs(alpha) > 1.0:
            turn_cmd = float(np.clip(alpha * 2.0, -self.MAX_TURN_SPEED, self.MAX_TURN_SPEED))
            return 0.0, turn_cmd, True

        nominal_turn = float(np.clip(2.0 * cruise_speed * math.sin(alpha) / max(distance, 0.1),
                                     -self.MAX_TURN_SPEED, self.MAX_TURN_SPEED))

        # Always run at normal speed (ramped by acceleration limit)
        forward_speed = min(cruise_speed, self._command_speed + self.max_acceleration * dt)

        # Test candidate turns at normal speed
        turns = np.unique(np.append(
            np.linspace(-self.MAX_TURN_SPEED, self.MAX_TURN_SPEED, 13),
            nominal_turn))
        speeds = np.full(len(turns), forward_speed)

        initial_speed = max(self._command_speed, abs(self._observed_forward_speed))
        initial_turn = (self._observed_turn if abs(self._observed_turn) > 0.03
                        else self._command_turn)

        obstacles = self._local_obstacles(world_pose, 1.2)
        margin = self.safety_margin
        trajectories, horizon_index = self._rollouts(
            speeds, turns, initial_speed, initial_turn)
        safe, clearance = self._collision_scores(trajectories, world_pose, obstacles, margin)

        # Follow the requested route whenever its nominal arc clears the body.
        # Extra clearance must not reward an unnecessary turn in open space.
        nominal_index = int(np.argmin(np.abs(turns - nominal_turn)))
        if safe[nominal_index]:
            return float(speeds[nominal_index]), nominal_turn, True

        if not np.any(safe):
            # No safe forward arc at normal speed: path is blocked ahead
            return 0.0, 0.0, False

        safe_indices = np.flatnonzero(safe)
        turn_diff = np.abs(turns[safe_indices] - nominal_turn)
        scores = -2.0 * turn_diff + 0.5 * clearance[safe_indices]
        best = safe_indices[int(np.argmax(scores))]
        return float(speeds[best]), float(turns[best]), True

    def _has_progress(self, pose, now):
        if self._progress_anchor is None:
            self._progress_anchor = pose
            self._progress_time = now
        previous = self._progress_anchor
        moved = math.hypot(pose[0] - previous[0], pose[1] - previous[1])
        turned = abs(math.remainder(pose[2] - previous[2], math.tau))
        if moved > 0.04 or turned > 0.1:
            self._progress_anchor = pose
            self._progress_time = now
        commanded = (abs(self._command_speed) > 0.05
                     or abs(self._command_turn) > 0.15)
        return not commanded or now - self._progress_time < self.stall_timeout

    def run_step(self):
        now = self._now()
        self._update_body_geometry_from_tf()
        healthy = (self.pose is not None
                   and self.mapdata is not None and self._scan_valid
                   and self._localization_valid and not self._wheel_slip
                   and now - self._last_scan_time <= self.scan_timeout
                   and now - self._last_odom_time <= self.odom_timeout
                   and now - self._last_localization_time <= self.localization_timeout
                   and now - self._last_heartbeat_time <= self.navigator_timeout
                   and self._map_frame == self._frame_id)

        # Active reverse recovery
        if self._reversing:
            if now - self._reverse_start_time < self._reverse_duration:
                self.send_speed(self._reverse_speed, 0.0)
                return
            else:
                self._reversing = False
                self.stop()
                self._progress_anchor = None
                self._progress_time = now
                self._stall_latched = False
                self._publish_recovery_needed(False, now)
                self._clear_blocked(now)
                self._publish_blocked(True, now)
                self.get_logger().info('Reverse recovery complete; cleared blocked and requested replan.')
                return

        # Start recovery lease from navigator
        if self._start_recovery_requested:
            self._start_reverse(now, reason='navigator recovery lease')
            return

        if not healthy:
            self.stop()
            return

        if not self.path.poses:
            self._publish_recovery_needed(False, now)
            self.stop()
            self._clear_blocked(now)
            return

        healthy = (healthy and self._path_valid and self._path_frame == self._frame_id)
        if not healthy:
            self.stop()
            return

        world_pose = (self.pose.position.x, self.pose.position.y,
                      euler_from_quaternion(self.pose.orientation))
        goal_distance = math.hypot(world_pose[0] - self._path_xy[-1, 0],
                                   world_pose[1] - self._path_xy[-1, 1])
        if goal_distance < self.DISTANCE_TOLERANCE:
            self.stop()
            self._clear_blocked(now)
            return

        # Physical stall detection
        if self._stall_latched or not self._has_progress(world_pose, now):
            self._stall_latched = True
            self.get_logger().warning('Commanded motion made no progress; reversing.')
            self._start_reverse(now, reason='stall')
            return

        target, path_ahead = self._target(world_pose[0], world_pose[1])
        dt = (self._control_period if self._last_step_time is None
              else min(self._control_period * 2, max(0.001, now - self._last_step_time)))
        self._last_step_time = now

        forward, turn, viable = self._choose_command(
            target, path_ahead, world_pose, now - self._last_scan_time, dt)

        if viable:
            self._publish_recovery_needed(False, now)
            self._clear_blocked(now)
            self.send_speed(forward, turn)
        else:
            self.get_logger().warning('No safe forward trajectory (blocked ahead); reversing.')
            self._start_reverse(now, reason='blocked ahead')

def main(args=None):
    # Publish the final zero command before shutting down the ROS context.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = PurePursuit()

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop()
            node.cmd_vel.wait_for_all_acked(Duration(seconds=0.1))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
