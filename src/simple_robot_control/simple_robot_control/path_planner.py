"""Drive the robot to reachable frontiers while checking live LiDAR returns."""

import math
import time
from concurrent.futures import ThreadPoolExecutor

from geometry_msgs.msg import Point, PoseStamped, Twist

from nav_msgs.msg import OccupancyGrid, Odometry, Path

import numpy as np

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time

from sensor_msgs.msg import LaserScan

from std_msgs.msg import Bool

from tf2_ros import Buffer, TransformException, TransformListener

from visualization_msgs.msg import Marker

from .frontier_explorer import (FrontierMemory, Grid, frontier_is_productive,
                                frontier_points, motion_is_clear, plan_frontier)
from .lidar_odom import apply_T, make_T, scan_to_points
from .pure_pursuit import command_for_path


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class AutonomousExplorer(Node):
    """Combine the occupancy map, EKF pose and timestamped obstacle returns."""

    def __init__(self):
        super().__init__('path_planner')
        self.max_speed = self.declare_parameter('max_speed', 2.0).value
        self.max_turn = self.declare_parameter('max_turn', 1.4).value
        if not all(math.isfinite(value) and value > 0.0 for value in (
                self.max_speed, self.max_turn)):
            raise ValueError('Speed limits must be finite and positive')

        self.grid = self.pose = self.hits = self.pending_scan = None
        self.grid_time = self.pose_time = self.scan_time = None
        self.tracking_time = None
        self.tracking_valid = None
        self.scan_problem = 'Waiting for /lidar_2d_scan.'
        self.path = None
        self.plan = None
        self.start_pose = None
        self.returning_home = False
        self.home_reached = False
        self.frontier_memory = FrontierMemory()
        self.replan_goal = None
        self.discard_plan = False
        self.next_plan = 0.0
        self.status = None
        self.planner = ThreadPoolExecutor(max_workers=1)
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._no_path_count = 0
        self._blocked_count = 0
        self._spin_start_time = None
        self._spin_duration = 14.0
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE)
        visualization_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)
        self.create_subscription(Odometry, '/odom_fused', self._on_pose, 1)
        self.create_subscription(LaserScan, '/lidar_2d_scan', self._on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Bool, '/localization/tracking_valid', self._on_tracking, 1)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.path_pub = self.create_publisher(
            Path, '/exploration/path', visualization_qos)
        self.frontier_pub = self.create_publisher(
            Marker, '/exploration/frontiers', visualization_qos)
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.05, self._tick, clock=self.steady_clock)
        # Full-map marker extraction must not run on every control tick.
        self.create_timer(0.5, self._publish_frontiers, clock=self.steady_clock)

    def _on_map(self, msg):
        info = msg.info
        origin = (info.origin.position.x, info.origin.position.y)
        q = info.origin.orientation
        if (msg.header.frame_id != 'odom' or not math.isfinite(info.resolution)
                or info.resolution <= 0.0 or not all(map(math.isfinite, origin))
                or (q.x, q.y, q.z) != (0.0, 0.0, 0.0)
                or not info.height or not info.width
                or len(msg.data) != info.height * info.width):
            self.grid = None
            return
        values = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
        self.grid = Grid(values, origin, info.resolution)
        self.grid_time = self._received(msg.header.stamp)

    def _on_pose(self, msg):
        pose = msg.pose.pose
        self.pose = (pose.position.x, pose.position.y, _yaw(pose.orientation))
        if (msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_footprint'
                or not all(map(math.isfinite, self.pose))):
            self.pose = None
            return
        received = self._received(msg.header.stamp)
        if self.pose_time is not None and received[0] < self.pose_time[0]:
            # A simulation clock reset invalidates the previous map and path.
            self.grid = self.hits = self.pending_scan = None
            # Replace the object so an old planning job cannot restore it.
            self.frontier_memory = FrontierMemory()
            self.replan_goal = None
            self._clear_path()
        self.pose_time = received

    def _on_scan(self, msg):
        self.pending_scan = msg, self._received(msg.header.stamp)

    def _on_tracking(self, msg):
        self.tracking_time = time.monotonic()
        self.tracking_valid = bool(msg.data)
        if not self.tracking_valid:
            self._command()

    @staticmethod
    def _received(stamp):
        return Time.from_msg(stamp).nanoseconds, time.monotonic()

    @staticmethod
    def _fresh(received, now, wall_now, timeout):
        return (received is not None and wall_now - received[1] <= timeout
                and -0.10 <= (now - received[0]) * 1e-9 <= timeout)

    def _input_problem(self, now, wall_now):
        """Name the input preventing motion without weakening freshness checks."""
        if self.tracking_valid is False:
            return 'Waiting for localization: EKF reports tracking invalid.'
        if self.tracking_valid is None or self.tracking_time is None:
            return 'Waiting for /localization/tracking_valid.'
        if wall_now - self.tracking_time > 0.6:
            return 'Waiting for /localization/tracking_valid updates (0.6 s timeout).'
        inputs = (
            ('/odom_fused', self.pose, self.pose_time, 0.5),
            ('/lidar_2d_scan', self.hits, self.scan_time, 0.5),
            ('/map', self.grid, self.grid_time, 2.0),
        )
        for topic, value, received, timeout in inputs:
            if value is None:
                if topic == '/lidar_2d_scan':
                    return self.scan_problem
                return f'Waiting for valid {topic} data.'
            if received is None or wall_now - received[1] > timeout:
                return f'Waiting for {topic} updates ({timeout:g} s wall-time timeout).'
            age = (now - received[0]) * 1e-9
            if age < -0.10:
                return f'Waiting for /clock to reach the {topic} timestamp.'
            if age > timeout:
                return f'Waiting for {topic} within {timeout:g} s of /clock.'
        return None

    def _update_obstacles(self, now, wall_now):
        if self.pending_scan is None:
            return
        msg, received = self.pending_scan
        if not self._fresh(received, now, wall_now, 0.5):
            self.pending_scan = None
            self.scan_problem = 'Waiting for a fresh /lidar_2d_scan (0.5 s timeout).'
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', msg.header.frame_id, Time.from_msg(msg.header.stamp)).transform
        except TransformException:
            # Wait for the EKF transform to reach the scan's acquisition time.
            self.scan_problem = (
                f'Waiting for LiDAR transform: {msg.header.frame_id} to odom '
                'at the scan timestamp.')
            return
        self.pending_scan = None
        self.scan_time = received
        if not all(map(math.isfinite, (msg.range_min, msg.range_max,
                                       msg.angle_min, msg.angle_increment))):
            self.hits = None
            self.scan_problem = 'Waiting for /lidar_2d_scan with valid metadata.'
            return
        points = scan_to_points(msg, np.nextafter(msg.range_max, -np.inf))
        self.hits = None
        if len(points):
            mount = transform.translation
            self.hits = apply_T(make_T(mount.x, mount.y, _yaw(transform.rotation)), points)
        else:
            self.scan_problem = 'Waiting for /lidar_2d_scan with usable range returns.'

    def _publish_path(self):
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        if self.path is not None:
            for x, y in self.path:
                point = PoseStamped()
                point.header = msg.header
                point.pose.position.x = float(x)
                point.pose.position.y = float(y)
                point.pose.orientation.w = 1.0
                msg.poses.append(point)
        self.path_pub.publish(msg)

    def _publish_frontiers(self):
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'exploration'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.1
        marker.color.a = 1.0
        if self.grid is not None and self.pose is not None and self.hits is not None:
            marker.points = [
                Point(x=float(x), y=float(y), z=0.08)
                for x, y in frontier_points(self.grid, self.pose, self.hits)
            ]
        self.frontier_pub.publish(marker)

    def _clear_path(self):
        if self.path is not None:
            self.path = None
            self._publish_path()
        self.discard_plan = True
        self._no_path_count = 0
        self._blocked_count = 0
        self._spin_start_time = None

    def _update_path(self, wall_now):
        if self.plan is not None and self.plan.done():
            if not self.discard_plan:
                new_path = self.plan.result()
                obsolete = (new_path is not None and not self.returning_home
                            and not frontier_is_productive(self.grid, new_path[-1]))
                if obsolete:
                    self._set_status(
                        'Frontier goal was mapped during planning; selecting another.')
                    self.next_plan = wall_now
                else:
                    if new_path is None:
                        self._no_path_count += 1
                    else:
                        self._no_path_count = 0
                        self._spin_start_time = None
                    if new_path is not None or self.path is None:
                        self.path = new_path
                        self._publish_path()
            self.plan = None
        if self.path is not None or self.plan is not None or wall_now < self.next_plan:
            return
        if self.returning_home:
            previous_goal = None
            target_goal = self.start_pose[:2]
        elif self.replan_goal is not None:
            previous_goal = None
            target_goal = self.replan_goal
            self.replan_goal = None
        else:
            previous_goal = self.frontier_memory.goal
            target_goal = None
        self.plan = self.planner.submit(
            plan_frontier, self.grid, self.pose, self.hits, previous_goal,
            target_goal, self.frontier_memory)
        self.discard_plan = False
        self.next_plan = wall_now + 0.6

    def _command(self, speed=0.0, turn=0.0):
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(turn)
        self.cmd_pub.publish(cmd)

    def _drive(self):
        wall_now = time.monotonic()
        if not self.returning_home and not frontier_is_productive(
                self.grid, self.path[-1]):
            self._set_status('Frontier goal has been mapped; selecting another.')
            self._clear_path()
            self.next_plan = wall_now
            self._command()
            return
        distances = np.linalg.norm(self.path - self.pose[:2], axis=1)
        if distances[-1] <= 0.15:
            if self.returning_home:
                self.home_reached = True
                self._set_status('Returned to the starting position.')
            else:
                self._set_status('Frontier goal reached; selecting another.')
            self._clear_path()
            self._command()
            return
        command = command_for_path(
            self.path, self.pose, self.max_speed, self.max_turn,
            self.grid, self.hits)
        if command is not None:
            self._set_status('Returning to the starting position.' if self.returning_home
                             else 'Exploring a reachable frontier.')
            self._command(*command)
            self._blocked_count = 0
            return
        self._blocked_count += 1
        if self._blocked_count >= 3:
            self._set_status('Route blocked by live collision checks; replanning the goal.')
            self.replan_goal = tuple(self.path[-1])
            self._clear_path()
            self.next_plan = wall_now
        else:
            self._set_status('Route blocked by live collision checks; stopping.')
        self._command()

    def _set_status(self, status):
        if status != self.status:
            self.get_logger().info(status)
            self.status = status

    def _tick(self):
        wall_now = time.monotonic()
        now = self.get_clock().now().nanoseconds
        self._update_obstacles(now, wall_now)
        problem = self._input_problem(now, wall_now)
        if problem is not None:
            self._command()
            # Pause motion, preserving the goal, path and planning job. A brief
            # data gap must not discard the route on every recovery cycle.
            self._set_status(problem)
            return
        if self.start_pose is None:
            self.start_pose = self.pose
        self._update_path(wall_now)
        if self.home_reached:
            self._command()
            self._set_status('Returned to the starting position.')
            return
        if self.path is None:
            if self.returning_home:
                self._command()
                status = ('Planning a route to the starting position.'
                          if self.plan is not None
                          else 'No safe route to the starting position; retrying.')
                self._set_status(status)
                return
            if (self.plan is not None and self._spin_start_time is None
                    and self._no_path_count <= 10):
                self._command()
                self._set_status('Planning a route to a frontier.')
                return
            if self._no_path_count > 10:
                if self._spin_start_time is None:
                    self._spin_start_time = wall_now
                elapsed = wall_now - self._spin_start_time
                if elapsed < self._spin_duration:
                    turn_speed = 0.5
                    if motion_is_clear(self.grid, self.pose, self.hits, 0.0, turn_speed):
                        self._command(0.0, turn_speed)
                    else:
                        self._command(0.0, 0.0)
                    self._set_status('Scanning surroundings for frontiers.')
                    return
                else:
                    self.returning_home = True
                    self.replan_goal = None
                    # A frontier job can still be running during the scan.
                    # Its result must not be mistaken for a route home.
                    self._clear_path()
                    self.next_plan = wall_now
                    self._set_status('Returning to the starting position.')
                    return
            self._command()
            self._set_status('Looking for a reachable frontier.')
            return
        self._no_path_count = 0
        self._spin_start_time = None
        self._drive()


def main(args=None):
    """Run exploration and stop the robot when the node shuts down."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = AutonomousExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._command()
        node.planner.shutdown(wait=True, cancel_futures=True)
        node.destroy_node()
        rclpy.try_shutdown()
