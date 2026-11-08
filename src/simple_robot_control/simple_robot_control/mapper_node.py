#!/usr/bin/env python3
# Authors: Akash Mohapatra, Manas Chintawar

import copy
import math
import time
from collections import deque

import cv2

from nav_msgs.msg import OccupancyGrid, Odometry

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from sensor_msgs.msg import LaserScan

from std_msgs.msg import Bool

from tf2_ros import Buffer, TransformException, TransformListener


class SimpleMapperNode(Node):

    def __init__(self):
        super().__init__('simple_mapper')

        self.resolution = self.declare_parameter('resolution', 0.05).value
        self.width = self.declare_parameter('width', 600).value
        self.height = self.declare_parameter('height', 600).value
        self.base_frame = self.declare_parameter(
            'base_frame', 'base_footprint').value
        self.pose_topic = self.declare_parameter(
            'pose_topic', '/lidar_odom').value
        self.tracking_timeout = self.declare_parameter(
            'tracking_timeout', 0.75).value
        self._tracking_valid = False
        self._tracking_received = None
        self.MAX_TRUSTED_RANGE = self.declare_parameter(
            'max_range', 10.0).value
        self.beam_stride = self.declare_parameter('beam_stride', 2).value
        if (self.resolution <= 0 or self.width <= 0 or self.height <= 0
                or self.MAX_TRUSTED_RANGE <= 0 or self.beam_stride < 1):
            raise ValueError(
                'Map dimensions, range and beam stride must be positive')

        shape = (self.height, self.width)
        self.log_odds = np.zeros(shape, dtype=np.float32)
        self._free_rays = np.zeros(shape, dtype=np.uint8)
        self._hit_cells = np.zeros(shape, dtype=np.uint8)

        # Every occupied cell remains correctable by later clear observations.
        # Cap confidence so a misplaced wall can be cleared in a few scans.
        self.L_OCC = 0.85
        self.L_FREE = -0.4
        self.MAX_LOG_ODDS = 3.0
        self.MIN_LOG_ODDS = -2.0
        self._occupied_threshold = math.log(0.65 / 0.35)
        self._free_threshold = math.log(0.35 / 0.65)

        self.grid_msg = OccupancyGrid()
        self.grid_msg.header.frame_id = 'odom'
        self.grid_msg.info.resolution = self.resolution
        self.grid_msg.info.width = self.width
        self.grid_msg.info.height = self.height
        self.grid_msg.info.origin.position.x = (
            -self.width * self.resolution / 2)
        self.grid_msg.info.origin.position.y = (
            -self.height * self.resolution / 2)
        self.grid_msg.info.origin.orientation.w = 1.0
        self.inflated_grid_msg = copy.deepcopy(self.grid_msg)

        # Inflation belongs only to the navigation map, never the raw map.
        self.obstacle_merge_dist_m = 0.25
        merge_size = int(self.obstacle_merge_dist_m / self.resolution) | 1
        self.merge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (merge_size, merge_size))
        self.inflation_radius_m = 0.15
        inflate_size = 2 * math.ceil(
            self.inflation_radius_m / self.resolution) + 1
        self.inflation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (inflate_size, inflate_size))

        self._odom_history = deque(maxlen=500)
        self._pending_scans = deque(maxlen=10)
        self._last_scan_ns = None
        self._scan_geometry = None
        self._mount_frame = None
        self._mount_pose = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)
        self.create_subscription(
            Odometry, self.pose_topic, self.odom_callback, 10)
        self.create_subscription(
            Bool, '/localization/tracking_valid', self.tracking_callback, 1)
        self.create_subscription(
            LaserScan, '/lidar_2d_scan', self.scan_callback, sensor_qos)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.inflated_map_pub = self.create_publisher(
            OccupancyGrid, '/map_inflated', 1)
        self.publish_counter = 0
        self.publish_rate = 3
        self.get_logger().info(
            'Mapper started with scan-time poses and correctable occupancy.')

    @staticmethod
    def _stamp_ns(stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    @staticmethod
    def _yaw(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y),
                          1 - 2 * (q.y * q.y + q.z * q.z))

    def tracking_callback(self, msg):
        self._tracking_valid = bool(msg.data)
        self._tracking_received = time.monotonic()
        if not self._tracking_valid:
            self._pending_scans.clear()

    def _tracking_ready(self):
        return (self._tracking_valid and self._tracking_received is not None
                and time.monotonic() - self._tracking_received
                <= self.tracking_timeout)

    def odom_callback(self, msg):
        if (msg.header.frame_id
                and msg.header.frame_id != self.grid_msg.header.frame_id):
            self.get_logger().warning(
                'Ignoring odometry in a different map frame.',
                throttle_duration_sec=5.0)
            return
        variances = np.asarray(msg.pose.covariance)[[0, 7, 35]]
        if (not np.isfinite(variances).all()
                or np.any(variances < 0) or np.any(variances >= 1e3)):
            # Failed/bootstrap registration deliberately carries huge noise.
            # It must never turn wheel fallback into newly painted map cells.
            return
        pose = msg.pose.pose
        sample = (pose.position.x, pose.position.y,
                  self._yaw(pose.orientation))
        if not all(math.isfinite(value) for value in sample):
            return
        stamp_ns = self._stamp_ns(msg.header.stamp)
        if self._odom_history and stamp_ns < self._odom_history[-1][0]:
            if self._odom_history[-1][0] - stamp_ns < 1_000_000_000:
                return
            # Simulation reset: old scans and occupied cells belong to the
            # previous timeline and must not overlap the newly spawned scene.
            self._odom_history.clear()
            self._pending_scans.clear()
            self._last_scan_ns = None
            self.log_odds.fill(0)
            self.publish_counter = 0
        if self._odom_history and stamp_ns == self._odom_history[-1][0]:
            self._odom_history.pop()
        self._odom_history.append((stamp_ns, sample))
        while (self._pending_scans
               and self._stamp_ns(self._pending_scans[0].header.stamp)
               <= stamp_ns):
            self.scan_callback(self._pending_scans.popleft())

    def _pose_at(self, stamp_ns):
        """Use the accepted registration for this exact sensor acquisition."""
        for sample_ns, pose in reversed(self._odom_history):
            if sample_ns == stamp_ns:
                return pose
            if sample_ns < stamp_ns:
                break
        return None

    def _sensor_mount(self, frame):
        if not frame:
            self.get_logger().warning(
                'Ignoring scan with no sensor frame.',
                throttle_duration_sec=5.0)
            return None
        if frame == self.base_frame:
            return 0.0, 0.0, 0.0
        if frame != self._mount_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame, frame, Time()).transform
            except TransformException:
                self.get_logger().warning(
                    f'Waiting for transform {self.base_frame} -> {frame}.',
                    throttle_duration_sec=5.0)
                return None
            self._mount_pose = (transform.translation.x,
                                transform.translation.y,
                                self._yaw(transform.rotation))
            self._mount_frame = frame
        return self._mount_pose

    def scan_callback(self, msg):
        if not self._tracking_ready():
            self._pending_scans.clear()
            return
        stamp_ns = self._stamp_ns(msg.header.stamp)
        if self._last_scan_ns is not None and stamp_ns <= self._last_scan_ns:
            return
        if not self._odom_history or stamp_ns > self._odom_history[-1][0]:
            if (not self._pending_scans or stamp_ns > self._stamp_ns(
                    self._pending_scans[-1].header.stamp)):
                # Bounded backlog permits scan registration to finish after
                # the next cloud arrives without starving map updates.
                self._pending_scans.append(msg)
            return
        pose = self._pose_at(stamp_ns)
        if pose is None:
            # Older than the history: never substitute the latest pose.
            return
        mount = self._sensor_mount(msg.header.frame_id)
        if mount is None:
            return
        if not self._integrate_scan(pose, mount, msg):
            return
        self._last_scan_ns = stamp_ns
        self.grid_msg.header.stamp = msg.header.stamp
        self.publish_counter += 1
        if self.publish_counter >= self.publish_rate:
            self.publish_counter = 0
            self.publish_map()

    def _integrate_scan(self, pose, mount, msg):
        """Add one independent hit/free observation per cell in this scan."""
        if (not math.isfinite(msg.range_max) or msg.range_max <= 0
                or not math.isfinite(msg.angle_min)
                or not math.isfinite(msg.angle_increment)):
            return False
        rx, ry, yaw = pose
        mx, my, mount_yaw = mount
        c, s = math.cos(yaw), math.sin(yaw)
        sx, sy = rx + c * mx - s * my, ry + s * mx + c * my
        origin = self.grid_msg.info.origin.position
        rx_grid = math.floor((sx - origin.x) / self.resolution)
        ry_grid = math.floor((sy - origin.y) / self.resolution)
        if not (0 <= rx_grid < self.width and 0 <= ry_grid < self.height):
            return False

        ranges = np.asarray(msg.ranges)
        geometry = (len(ranges), msg.angle_min, msg.angle_increment,
                    self.beam_stride)
        if geometry != self._scan_geometry:
            self._beam_indices = np.arange(0, len(ranges), self.beam_stride)
            angles = msg.angle_min + self._beam_indices * msg.angle_increment
            self._beam_cos = np.cos(angles)
            self._beam_sin = np.sin(angles)
            self._scan_geometry = geometry
        ranges = ranges[self._beam_indices]
        # An empty height-filtered bin is unknown, not a clear ray to 10 m.
        valid = (np.isfinite(ranges)
                 & (ranges >= max(msg.range_min, 0.0)) & (ranges > 0.0)
                 & (ranges < msg.range_max)
                 & (ranges <= self.MAX_TRUSTED_RANGE))
        if not np.any(valid):
            return False
        distances = ranges[valid]
        c, s = math.cos(yaw + mount_yaw), math.sin(yaw + mount_yaw)
        beam_cos, beam_sin = self._beam_cos[valid], self._beam_sin[valid]
        hit_x = np.floor((sx + distances * (c * beam_cos - s * beam_sin)
                          - origin.x) / self.resolution).astype(np.int32)
        hit_y = np.floor((sy + distances * (s * beam_cos + c * beam_sin)
                          - origin.y) / self.resolution).astype(np.int32)

        self._free_rays.fill(0)
        self._hit_cells.fill(0)
        start = (rx_grid, ry_grid)
        for x, y in zip(hit_x, hit_y):
            end = (int(x), int(y))
            visible, clipped_start, clipped_end = cv2.clipLine(
                (0, 0, self.width, self.height), start, end)
            if visible:
                cv2.line(self._free_rays, clipped_start, clipped_end, 1, 1)
            if 0 <= x < self.width and 0 <= y < self.height:
                self._hit_cells[y, x] = 1

        occupied = self._hit_cells != 0
        free = (self._free_rays != 0) & ~occupied
        self.log_odds[free] += self.L_FREE
        self.log_odds[occupied] += self.L_OCC
        np.clip(self.log_odds, self.MIN_LOG_ODDS,
                self.MAX_LOG_ODDS, out=self.log_odds)
        return True

    def publish_map(self):
        occupied = (self.log_odds > self._occupied_threshold).astype(np.uint8)
        free = self.log_odds < self._free_threshold
        merged = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, self.merge_kernel)
        inflated = cv2.dilate(merged, self.inflation_kernel, iterations=1)

        raw_map = np.full(self.log_odds.shape, -1, dtype=np.int8)
        raw_map[free] = 0
        raw_map[occupied != 0] = 100
        inflated_map = np.full(self.log_odds.shape, -1, dtype=np.int8)
        inflated_map[free] = 0
        inflated_map[inflated != 0] = 100

        self.grid_msg.data = raw_map.ravel().tolist()
        self.map_pub.publish(self.grid_msg)
        self.inflated_grid_msg.header.stamp = self.grid_msg.header.stamp
        self.inflated_grid_msg.data = inflated_map.ravel().tolist()
        self.inflated_map_pub.publish(self.inflated_grid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
