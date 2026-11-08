#!/usr/bin/env python3
# Authors: Akash Mohapatra, Manas Chintawar

import math
import time
from dataclasses import dataclass, field

import cv2

from geometry_msgs.msg import Point, PoseStamped

from nav_msgs.msg import OccupancyGrid, Odometry, Path

import numpy as np

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from simple_robot_control.path_planner import PathPlanner

from std_msgs.msg import Bool, Empty

from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Frontier:
    size: int
    centroid: Point
    cells: np.ndarray | None = field(default=None, repr=False, compare=False)
    approach: Point | None = None
    status: str = 'blocked'


@dataclass
class FrontierExclusion:
    point: Point
    expires: float
    signature: bytes


class FrontierExplorer(Node):

    # ── tuning ──────────────────────────────────────────────────────
    # metres; margin to mark a goal as reached (must be > grid resolution)
    GOAL_REACHED_DIST = 0.35
    # in seconds; Min cooldown time to look for new goals
    REPLAN_COOLDOWN_S = 2.0
    # Min number of contiguous edge cells to make a valid frontier
    MIN_FRONTIER_SIZE = 8
    # After max explore failures it concludes that the map is fully explored
    # and returns home
    NUM_EXPLORE_FAILS = 15
    # Bound each shared search; try the next batch if a route is unavailable.
    PLAN_BATCH_SIZE = 5
    FRONTIER_SPAN_M = 1.0
    FRONTIER_REACHED_DIST = 0.30
    VISITED_RADIUS_M = 0.30
    APPROACH_RANGE_M = 0.65
    BLOCKED_RETRY_S = 30.0
    # seconds of NO PROGRESS before giving up (path-index based)
    GOAL_TIMEOUT_S = 15.0
    # Actual minimum inflation is derived from robot geometry and resolution.
    DEFAULT_KERNEL_SIZE = 1
    # ────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('frontier_explorer')

        self.robot_length = self.declare_parameter('robot_length', 0.46).value
        self.robot_width = self.declare_parameter('robot_width', 0.34).value
        self.safety_margin = self.declare_parameter(
            'safety_margin', 0.06).value
        self._localization_valid = False
        self._localization_received = None
        self._localization_paused = False
        self._runtime_replan = False
        self._last_runtime_replan = -math.inf
        self._blocked_goal = None
        self._blocked_position = None
        self._blocked_retries = 0
        self._recovering_start = False
        self._local_recovery_needed = False
        self._local_recovery_received = -math.inf
        self._recovery_failed = False

        self.path_pub = self.create_publisher(Path, '/pure_pursuit/path', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/frontiers_vis', QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.heartbeat_pub = self.create_publisher(
            Empty, '/navigation/heartbeat', 1)
        self.recovery_pub = self.create_publisher(
            Bool, '/navigation/recover_start', 1)

        self.create_subscription(
            Odometry, '/odom_fused', self.odom_callback, 10)
        self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_pose_callback, 10)
        self.create_subscription(
            Bool, '/navigation/blocked', self.blocked_callback, 1)
        self.create_subscription(
            Bool, '/navigation/recovery_needed', self.recovery_needed_callback, 1)
        self.create_subscription(
            Bool, '/navigation/recovery_failed', self.recovery_failed_callback, 1)
        self.create_subscription(
            Bool, '/localization/tracking_valid', self.tracking_callback, 1)

        self.pose = None
        self.mapdata = None
        self._map_array = None
        self._map_geometry = None
        self._cost_map = None
        self._cspace_array = None
        self._cspace_kernel = None
        self._planning_grid = None
        self._planning_kernel = None
        self._free_mask = None
        self._free_labels = None
        self._frontier_cache = {}
        self._path_array = np.empty((0, 2), dtype=np.int32)

        # ── state management ────────────────────────────────────────
        self.initial_pose: Point | None = None
        self.returning_home = False

        self.current_goal_centroid: Point | None = None
        self.last_replan_time = self.get_clock().now()
        self.goal_start_time = self.get_clock().now()
        # reset when robot advances along path
        self.last_progress_time = time.monotonic()
        # furthest path cell index reached
        self.best_path_idx = 0
        self.no_frontiers_found_counter = 0
        self.is_finished_exploring = False
        # Frontier completion persists while accepting later manual commands.
        self.exploration_complete = False

        # --- Squeeze Mode State Variables ---
        self.return_home_fails = 0
        self.user_goal_fails = 0
        # Holds the reduced kernel size while driving
        self.active_kernel = self.DEFAULT_KERNEL_SIZE

        self.user_goal_centroid: Point | None = None
        self._manual_resume_exploration = False
        self._resume_frontier_centroid: Point | None = None

        self._frontier_exclusions = []
        # grid cells of the active path (for obstacle checking)
        self.current_path = []

        # Renew watchdog/recovery leases in wall time, even when Gazebo runs
        # below real time. Mission progress timestamps still use the ROS clock.
        self._timer_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(
            0.5, self.explore_loop, clock=self._timer_clock)
        self.heartbeat_timer = self.create_timer(
            0.25, self._publish_heartbeat, clock=self._timer_clock)
        self.marker_timer = self.create_timer(1.0, self._refresh_markers)
        self.get_logger().info('Frontier Explorer started.')

    # ── callbacks ───────────────────────────────────────────────────

    def _publish_heartbeat(self):
        self.heartbeat_pub.publish(Empty())

    def _request_start_recovery(self, active):
        # True is a short lease renewed by explore_loop, not a motion command.
        # The controller still checks sensors and each swept/braking footprint.
        if active or getattr(self, '_recovering_start', False):
            self.recovery_pub.publish(Bool(data=active))
        self._recovering_start = active

    def recovery_needed_callback(self, msg):
        self._local_recovery_needed = bool(msg.data)
        self._local_recovery_received = time.monotonic()

    def recovery_failed_callback(self, msg):
        if msg.data:
            self._recovery_failed = True

    def _begin_start_recovery(self):
        if not self._recovering_start:
            if self.current_goal_centroid is not None:
                self._resume_frontier_centroid = self.current_goal_centroid
                self.current_goal_centroid = None
            self.get_logger().warning(
                'Current pose needs clearance recovery; preserving the mission '
                'while the controller finds a checked escape.')
        self._recovery_failed = False
        self._runtime_replan = False
        self.current_path = []
        self._publish_stop()
        self._request_start_recovery(True)

    def tracking_callback(self, msg):
        self._localization_valid = bool(msg.data)
        self._localization_received = time.monotonic()
        if not self._localization_valid:
            self._request_start_recovery(False)
            self._localization_paused = True
            self._runtime_replan = False
            if self.current_path:
                self.current_path = []
                self._publish_stop()

    def _tracking_ready(self):
        return (self._localization_valid
                and self._localization_received is not None
                and time.monotonic() - self._localization_received <= 0.75)

    def _publish_stop(self):
        message = Path()
        message.header.frame_id = (
            self.mapdata.header.frame_id if self.mapdata is not None
            else 'odom') or 'odom'
        message.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(message)

    def blocked_callback(self, msg):
        """Cancel an unsafe route immediately and request a bounded replan."""
        now = time.monotonic()
        if (not msg.data or not self.current_path
                or now - self._last_runtime_replan < self.REPLAN_COOLDOWN_S):
            return
        self._last_runtime_replan = now
        self._runtime_replan = self._tracking_ready()
        self.current_path = []
        self._publish_stop()

    def _handle_runtime_replan(self):
        self._runtime_replan = False
        goal = (self.user_goal_centroid if self.user_goal_centroid is not None
                else self.initial_pose if self.returning_home
                else self.current_goal_centroid)
        position = self.pose.position
        same_goal = (goal is not None and self._blocked_goal is not None
                     and math.hypot(goal.x - self._blocked_goal.x,
                                    goal.y - self._blocked_goal.y) < 1.0)
        no_progress = (self._blocked_position is not None
                       and math.hypot(position.x - self._blocked_position.x,
                                      position.y - self._blocked_position.y)
                       < 0.25)
        if same_goal and no_progress:
            self._blocked_retries += 1
        else:
            self._blocked_goal = goal
            self._blocked_position = Point(x=position.x, y=position.y)
            self._blocked_retries = 1
        self.best_path_idx = 0
        if self._blocked_retries >= 3:
            self.get_logger().warning(
                'Repeated local blockage without progress; cancelling goal.')
            if self.user_goal_centroid is not None:
                self._finish_manual_goal()
                return False
            if self.returning_home:
                self.returning_home = False
                self.is_finished_exploring = True
                self._publish_stop()
                return False
            if goal is not None:
                self._exclude_frontier_goal(goal, reached=False)
            self._resume_frontier_centroid = None
        elif self.current_goal_centroid is not None:
            self._resume_frontier_centroid = self.current_goal_centroid
        self.current_goal_centroid = None
        return True

    def goal_pose_callback(self, msg: PoseStamped):
        self._request_start_recovery(False)
        self._blocked_retries = 0
        self._blocked_goal = None
        self._runtime_replan = False
        if self.user_goal_centroid is None:
            # Capture the mission phase only once, even if the user replaces
            # a manual waypoint before reaching it.
            if self.returning_home or self.is_finished_exploring:
                self.exploration_complete = True
            self._manual_resume_exploration = not self.exploration_complete
            if (self._manual_resume_exploration
                    and self.current_goal_centroid is not None):
                goal = self.current_goal_centroid
                self._resume_frontier_centroid = Point(
                    x=goal.x, y=goal.y, z=goal.z)
            elif not self._manual_resume_exploration:
                self._resume_frontier_centroid = None

        self.user_goal_centroid = msg.pose.position
        self.current_path = []
        self._publish_stop()

        # Reset fails and kernel when a new goal is clicked
        self.user_goal_fails = 0
        self.active_kernel = self.DEFAULT_KERNEL_SIZE

        self.get_logger().info(
            'Manual Override: Navigating to '
            f'({self.user_goal_centroid.x:.2f}, '
            f'{self.user_goal_centroid.y:.2f})')

        self.is_finished_exploring = False
        self.returning_home = False
        self.no_frontiers_found_counter = 0
        self.current_goal_centroid = None

    def _finish_manual_goal(self):
        """Restore the interrupted mission phase after any manual outcome."""
        self._request_start_recovery(False)
        self.current_path = []
        self._publish_stop()
        self.user_goal_centroid = None
        self.current_goal_centroid = None
        self.active_kernel = self.DEFAULT_KERNEL_SIZE
        self.user_goal_fails = 0
        self.return_home_fails = 0
        self.returning_home = False
        self.best_path_idx = 0
        self.no_frontiers_found_counter = 0
        resume = (self._manual_resume_exploration
                  and not self.exploration_complete)
        if resume:
            start = PathPlanner.world_to_grid(self.mapdata, self.pose.position)
            remaining = self.search_frontiers(start)
            self._evaluate_frontiers(
                remaining, self._prepare_planning_grid(), start)
            pending = bool(remaining) and any(
                math.isfinite(item.expires) for item in self._frontier_exclusions)
            if not any(f.approach is not None for f in remaining) and not pending:
                # The detour can finish the map, or arrive during the normal
                # no-frontier retry window. The manual waypoint is then the
                # final destination rather than a trigger for a trip home.
                self.exploration_complete = True
                resume = False
        self.is_finished_exploring = not resume
        self._manual_resume_exploration = False
        if resume:
            self.get_logger().info('Resuming frontier exploration.')
        else:
            self._resume_frontier_centroid = None
            self.get_logger().info(
                'Exploration is complete. Staying at the current position.')

    def _resume_frontier_path(self, frontiers, planner, start_cell):
        """Replan to an interrupted frontier if it remains reachable."""
        saved_goal = self._resume_frontier_centroid
        self._resume_frontier_centroid = None
        if saved_goal is None:
            return None, None
        nearby = [
            frontier for frontier in frontiers
            if frontier.approach is not None
            and math.hypot(frontier.approach.x - saved_goal.x,
                           frontier.approach.y - saved_goal.y) < 1.0
        ]
        if not nearby:
            return None, None
        frontier = min(nearby, key=lambda f: math.hypot(
            f.approach.x - saved_goal.x, f.approach.y - saved_goal.y))
        goal_cell = PathPlanner.world_to_grid(self.mapdata, frontier.approach)
        path, _, _, _ = planner.a_star(start_cell, goal_cell)
        if not path:
            return None, None
        return path, frontier.approach

    def _local_map_signature(self, point):
        """Only topology changes near a visited viewpoint make it worth retrying."""
        x, y = PathPlanner.world_to_grid(self.mapdata, point)
        radius = math.ceil(self.APPROACH_RANGE_M / self.mapdata.info.resolution)
        height, width = self._map_array.shape
        patch = self._map_array[max(0, y - radius):min(height, y + radius + 1),
                                max(0, x - radius):min(width, x + radius + 1)]
        return ((patch >= 50).astype(np.uint8)
                + 2 * (patch == -1).astype(np.uint8)).tobytes()

    def _exclude_frontier_goal(self, point, reached):
        self._frontier_exclusions.append(FrontierExclusion(
            point=point,
            expires=(math.inf if reached
                     else time.monotonic() + self.BLOCKED_RETRY_S),
            signature=self._local_map_signature(point)))

    def _prune_frontier_exclusions(self):
        now = time.monotonic()
        self._frontier_exclusions = [
            item for item in self._frontier_exclusions
            if item.expires > now
            and (math.isfinite(item.expires)
                 or item.signature == self._local_map_signature(item.point))
        ]

    def odom_callback(self, msg: Odometry):
        self.pose = msg.pose.pose
        if self.initial_pose is None:
            self.initial_pose = Point()
            self.initial_pose.x = self.pose.position.x
            self.initial_pose.y = self.pose.position.y
            self.initial_pose.z = self.pose.position.z

    def map_callback(self, msg: OccupancyGrid):
        if (msg.info.width <= 0 or msg.info.height <= 0
                or msg.info.resolution <= 0.0
                or len(msg.data) != msg.info.width * msg.info.height):
            self.get_logger().warning(
                'Ignoring an empty or invalid occupancy grid.')
            return
        grid = np.asarray(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        origin = msg.info.origin
        geometry = (
            msg.info.width, msg.info.height, msg.info.resolution,
            origin.position.x, origin.position.y,
            origin.orientation.x, origin.orientation.y,
            origin.orientation.z, origin.orientation.w, msg.header.frame_id)
        unchanged = (geometry == self._map_geometry
                     and np.array_equal(grid, self._map_array))
        if self._map_geometry is not None and geometry != self._map_geometry:
            # Grid cells of an existing route no longer identify the same
            # places.
            self.current_path = []
            self.current_goal_centroid = None
            self._frontier_exclusions.clear()
            self._publish_stop()
        self.mapdata = msg
        if unchanged:
            return
        self._map_array = grid
        self._map_geometry = geometry
        self._cost_map = None
        self._cspace_array = None
        self._cspace_kernel = None
        self._planning_grid = None
        self._planning_kernel = None
        self._free_mask = None
        self._free_labels = None
        self._frontier_cache = {}

    # ── frontier detection ──────────────────────────────────────────

    def search_frontiers(self, start_cell):
        """Label reachable free space and its unknown boundary in OpenCV."""
        grid = self._map_array
        if self._free_mask is None:
            self._free_mask = (grid >= 0) & (grid < 50)
        free = self._free_mask
        start_cell = self._nearest_walkable(free, start_cell)
        if start_cell is None:
            return []
        if self._free_labels is None:
            _, self._free_labels = cv2.connectedComponents(
                free.astype(np.uint8), connectivity=4)
        component = int(self._free_labels[start_cell[1], start_cell[0]])
        if component in self._frontier_cache:
            return self._frontier_cache[component]

        reachable = (self._free_labels == component).astype(np.uint8)
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        boundary = ((grid == -1) & (cv2.dilate(reachable, cross) != 0))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            boundary.astype(np.uint8), connectivity=8)
        frontiers = []
        span = max(1, round(self.FRONTIER_SPAN_M / self.mapdata.info.resolution))
        for label in range(1, count):
            x, y, width, height, size = stats[label]
            if size < self.MIN_FRONTIER_SIZE:
                continue
            ys, xs = np.nonzero(labels[y:y + height, x:x + width] == label)
            xs += x
            ys += y
            # One long/curved boundary needs several viewpoints. A single
            # centroid can sit behind a corner or hide an entire opening
            # after one visit. Apply the noise threshold to the whole edge,
            # then retain every cell in spatially bounded segments.
            cells = np.column_stack((xs, ys))
            _, groups = np.unique(cells // span, axis=0, return_inverse=True)
            for group in range(int(groups.max()) + 1):
                segment = cells[groups == group]
                center = segment.mean(axis=0)
                best = np.argmin(np.sum((segment - center) ** 2, axis=1))
                centroid = PathPlanner.grid_to_world(
                    self.mapdata, tuple(map(int, segment[best])))
                frontiers.append(Frontier(
                    size=len(segment), centroid=centroid, cells=segment))
        self._frontier_cache[component] = frontiers
        return frontiers

    def _evaluate_frontiers(self, frontiers, planner, start_cell):
        """Project every edge onto known, footprint-safe connected space."""
        self._prune_frontier_exclusions()
        reachable = planner.reachable_mask(start_cell)
        known_free = (self._map_array >= 0) & (self._map_array < 50)
        safe = reachable & known_free
        resolution = self.mapdata.info.resolution
        padding = math.ceil(self.APPROACH_RANGE_M / resolution)
        height, width = safe.shape
        robot = np.asarray(start_cell)
        for frontier in frontiers:
            frontier.approach = None
            frontier.status = 'blocked'
            cells = frontier.cells
            if cells is None or not len(cells):
                continue
            x0, y0 = np.maximum(cells.min(axis=0) - padding, 0)
            x1, y1 = np.minimum(cells.max(axis=0) + padding + 1,
                                (width, height))
            edge = np.ones((y1 - y0, x1 - x0), dtype=np.uint8)
            edge[cells[:, 1] - y0, cells[:, 0] - x0] = 0
            distances = cv2.distanceTransform(edge, cv2.DIST_L2, 5) * resolution
            ys, xs = np.nonzero(
                safe[y0:y1, x0:x1] & (distances <= self.APPROACH_RANGE_M))
            if not len(xs):
                continue
            offsets = distances[ys, xs]
            candidates = np.column_stack((xs + x0, ys + y0))
            # Already visible from the robot's current viewpoint; avoid
            # repeatedly issuing a goal that is immediately 'reached'.
            eligible = (np.linalg.norm(candidates - robot, axis=1) * resolution
                        > self.FRONTIER_REACHED_DIST + resolution)
            for item in self._frontier_exclusions:
                point = PathPlanner.world_to_grid(self.mapdata, item.point)
                eligible &= (np.linalg.norm(candidates - point, axis=1)
                             * resolution > self.VISITED_RADIUS_M)
            if not np.any(eligible):
                frontier.status = 'deferred'
                continue
            # Prefer a viewpoint just inside the edge, near this segment's
            # centre. If that is inflated/blocked, use another safe part.
            scores = (4.0 * np.abs(offsets - 0.12)
                      + 0.25 * resolution * np.linalg.norm(
                          candidates - cells.mean(axis=0), axis=1))
            scores[~eligible] = np.inf
            goal = tuple(map(int, candidates[np.argmin(scores)]))
            frontier.approach = PathPlanner.grid_to_world(self.mapdata, goal)
            frontier.status = 'reachable'

    @staticmethod
    def _nearest_walkable(walkable, cell, radius=14):
        """Find a free cell within the existing square search radius."""
        x, y = cell
        height, width = walkable.shape
        if 0 <= x < width and 0 <= y < height and walkable[y, x]:
            return cell
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        ys, xs = np.nonzero(walkable[y0:y1, x0:x1])
        if not len(xs):
            return None
        xs += x0
        ys += y0
        best = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        return int(xs[best]), int(ys[best])

    def _get_cspace_walkable(self):
        self._minimum_kernel = 5
        self.active_kernel = max(self.active_kernel, self._minimum_kernel)
        if (self._cspace_array is None
                or self._cspace_kernel != self.active_kernel):
            self._cspace_array = PathPlanner.calc_cspace_array(
                self.mapdata, kernel_size=self.active_kernel,
                grid=self._map_array)
            self._cspace_walkable = (
                (self._cspace_array >= 0) & (self._cspace_array < 50)
                & (self._map_array >= 0))
            self._cspace_kernel = self.active_kernel
        return self._cspace_walkable

    def _prepare_planning_grid(self):
        self._get_cspace_walkable()
        if self._cost_map is None:
            self._cost_map = PathPlanner.calc_cost_map(
                self.mapdata, grid=self._map_array)
        if (self._planning_grid is None
                or self._planning_kernel != self.active_kernel):
            self._planning_grid = PathPlanner.prepare(
                self.mapdata, self._cost_map, grid=np.where(
                    self._map_array >= 0, self._cspace_array, -1))
            self._planning_kernel = self.active_kernel
        return self._planning_grid

    @property
    def current_path(self):
        """Return the active route as grid cells."""
        return self._current_path

    @current_path.setter
    def current_path(self, path):
        """Replace the route and refresh its cached cell coordinates."""
        self._current_path = path
        self._path_array = np.asarray(path, dtype=np.int32).reshape(-1, 2)

    def _publish_path(self, path, real_start, start_cell):
        if not path:
            self.current_path = []
            self._publish_stop()
            return
        if start_cell != real_start or path[0] != real_start:
            path = [real_start] + list(path)
        self.current_path = path
        self.path_pub.publish(PathPlanner.path_to_message(
            self.mapdata, path, self.get_clock().now().to_msg()))

    # ── helpers ─────────────────────────────────────────────────────

    def _dist_to_point(self, p: Point) -> float:
        if self.pose is None:
            return float('inf')
        dx = self.pose.position.x - p.x
        dy = self.pose.position.y - p.y
        return math.hypot(dx, dy)

    def _is_path_blocked(self, nearest_idx=None, use_cspace=False) -> bool:
        """Check whether actual obstacles block the stored path ahead."""
        if not self.current_path or self.mapdata is None or self.pose is None:
            return False

        if nearest_idx is None:
            nearest_idx = self._get_nearest_path_idx()
        ahead = self._path_array[nearest_idx:]
        if not len(ahead):
            return False
        xs, ys = ahead[:, 0], ahead[:, 1]
        height, width = self._map_array.shape
        if np.any((xs < 0) | (xs >= width) | (ys < 0) | (ys >= height)):
            return True
        return bool(np.any(self._map_array[ys, xs] >= 50))

    def _is_at_end_of_path(self, tolerance=None) -> bool:
        """Check if the robot has reached the end of the stored path.
        Used to detect when a truncated path has been completed so we can
        replan with updated map data instead of waiting for goal timeout."""
        if not self.current_path or self.pose is None or self.mapdata is None:
            return False
        last_cell = self.current_path[-1]
        last_world = PathPlanner.grid_to_world(self.mapdata, last_cell)
        dist = math.hypot(
            self.pose.position.x - last_world.x,
            self.pose.position.y - last_world.y)
        return dist < (self.GOAL_REACHED_DIST if tolerance is None else tolerance)

    def _get_nearest_path_idx(self) -> int:
        """Return the index of the path cell closest to the robot."""
        if not self.current_path or self.pose is None or self.mapdata is None:
            return 0
        robot_cell = PathPlanner.world_to_grid(
            self.mapdata, self.pose.position)
        offsets = self._path_array - robot_cell
        return int(np.argmin(np.einsum('ij,ij->i', offsets, offsets)))

    # ── main loop ───────────────────────────────────────────────────

    def explore_loop(self):
        if (self.is_finished_exploring
                or self.pose is None or self.mapdata is None):
            self._request_start_recovery(False)
            return
        if not self._tracking_ready():
            self._request_start_recovery(False)
            if self.current_path:
                self.current_path = []
                self._publish_stop()
            self._localization_paused = True
            return
        if self._localization_paused:
            self._localization_paused = False
            self._runtime_replan = False
            if self.current_goal_centroid is not None:
                self._resume_frontier_centroid = self.current_goal_centroid
                self.current_goal_centroid = None
        real_start = PathPlanner.world_to_grid(
            self.mapdata, self.pose.position)
        start_cell = real_start

        # A locally unsafe footprint is a start-pose problem, even when its
        # cell passes circular inflation. Changing frontier goals cannot fix it.
        local_recovery = (self._local_recovery_needed
                          and time.monotonic() - self._local_recovery_received < 0.75)
        if self._recovery_failed:
            # The controller exhausted its recovery budget without escaping.
            # Abandon the current goal that led here, shrink the C-space
            # margin, and let the planner try a different frontier.
            self._recovery_failed = False
            self._request_start_recovery(False)
            goal = (self._resume_frontier_centroid
                    if self._resume_frontier_centroid is not None
                    else self.current_goal_centroid)
            if goal is not None:
                self._exclude_frontier_goal(goal, reached=False)
            self._resume_frontier_centroid = None
            self.current_goal_centroid = None
            self.current_path = []
            self._publish_stop()
            self.active_kernel = max(
                self._minimum_kernel, self.active_kernel - 2)
            self.get_logger().warning(
                'Recovery failed; excluding goal and replanning with reduced '
                f'C-space kernel {self.active_kernel}.')
            # Fall through to pick a new goal.
        elif self._recovering_start or local_recovery:
            planner = self._prepare_planning_grid()
            walkable_start = real_start if planner.is_walkable(real_start) else self._nearest_walkable(self._cspace_walkable, real_start)
            if (walkable_start is None or not planner.is_walkable(walkable_start) or local_recovery):
                self._begin_start_recovery()
                return
            self._request_start_recovery(False)
            self._blocked_retries = 0
            self._runtime_replan = False
            self.last_progress_time = time.monotonic()
            self.get_logger().info('Start clearance recovered; resuming route planning.')
        if self._runtime_replan and not self._handle_runtime_replan():
            return

        # --- Commit to current frontier goal ---
        # Check if we've reached/timed out the goal, or if path is blocked.
        # If goal is still valid and path is clear, skip expensive replan.
        if (self.user_goal_centroid is None
                and not self.returning_home
                and self.current_goal_centroid is not None):

            # Check: did we reach the goal?
            if (self._dist_to_point(self.current_goal_centroid)
                    < self.FRONTIER_REACHED_DIST):
                self.get_logger().info(
                    'Reached frontier viewpoint; evaluating the remaining edge.')
                self._exclude_frontier_goal(
                    self.current_goal_centroid, reached=True)
                self.current_goal_centroid = None
                self.current_path = []
                self._publish_stop()  # stop Pure Pursuit immediately
                # Fall through to pick a new goal

            # Check: is the robot making progress along the path?
            # Track which path cell the robot is nearest to; if the index
            # advances, the robot is following the path (even during U-turns
            # where Euclidean distance to the goal temporarily increases).
            else:
                current_idx = self._get_nearest_path_idx()
                # advanced by >2 cells
                if current_idx > self.best_path_idx + 2:
                    self.best_path_idx = current_idx
                    self.last_progress_time = time.monotonic()

                stall_seconds = time.monotonic() - self.last_progress_time

                if stall_seconds > self.GOAL_TIMEOUT_S:
                    self.get_logger().warning(
                        f'No path progress for {stall_seconds:.1f}s '
                        f'(stuck at path cell {self.best_path_idx}/'
                        f'{len(self.current_path)}). '
                        f'Deferring this viewpoint and replanning.')
                    self._exclude_frontier_goal(
                        self.current_goal_centroid, reached=False)
                    self.current_goal_centroid = None
                    self.current_path = []
                    self.best_path_idx = 0
                    # stop Pure Pursuit immediately
                    self._publish_stop()
                    # Fall through to pick a new goal

                # Check: is the path blocked by a new obstacle?
                elif self._is_path_blocked(current_idx, use_cspace=True):
                    self.get_logger().warning(
                        'Obstacle detected on path! Forcing immediate replan.')
                    self._resume_frontier_centroid = self.current_goal_centroid
                    self.current_goal_centroid = None
                    self.current_path = []
                    self.best_path_idx = 0
                    # stop Pure Pursuit immediately
                    self._publish_stop()
                    # Fall through to replan

                # Check: have we completed a truncated path? (blind-turn safe
                # approach)
                # The robot reached the end of the path but is still far from
                # the
                # actual goal — replan now with the freshly-mapped
                # surroundings.
                elif self._is_at_end_of_path(self.FRONTIER_REACHED_DIST):
                    self.get_logger().info(
                        'Reached end of route — '
                        'replanning with updated map.')
                    self.current_goal_centroid = None
                    self.current_path = []
                    self.best_path_idx = 0
                    # stop Pure Pursuit immediately
                    self._publish_stop()
                    # Fall through to replan

                else:
                    # Goal is active, path is clear, making progress — keep
                    # going
                    return

        # Check the actual requested target even when following a truncated
        # path.
        if self.user_goal_centroid is not None:
            raw_goal_cell = PathPlanner.world_to_grid(
                self.mapdata, self.user_goal_centroid)
            if (PathPlanner.is_cell_in_bounds(self.mapdata, raw_goal_cell)
                    and PathPlanner.get_cell_value(
                        self.mapdata, raw_goal_cell) >= 50):
                self.get_logger().error(
                    'Manual goal is inside a known obstacle! Rejecting.')
                self._finish_manual_goal()
                return

        # Manual/home routes can be followed until completion or a new
        # obstacle,
        # just like exploration routes, without running another search each
        # second.
        if ((self.user_goal_centroid is not None or self.returning_home)
                and self.current_path):
            if (not self._is_path_blocked(use_cspace=True)
                    and not self._is_at_end_of_path()):
                return
            self.current_path = []
            self._publish_stop()

        if self.user_goal_centroid is None and not self.returning_home:
            self.active_kernel = self.DEFAULT_KERNEL_SIZE
        planner = self._prepare_planning_grid()
        if not planner.is_walkable(real_start):
            walkable_start = self._nearest_walkable(self._cspace_walkable, real_start)
            if walkable_start is not None and planner.is_walkable(walkable_start):
                start_cell = walkable_start
            else:
                self._begin_start_recovery()
                return
        else:
            start_cell = real_start
        if self._recovering_start:
            self.get_logger().info(
                'Start clearance recovered; resuming route planning.')
        self._request_start_recovery(False)

        # ── MANUAL USER GOAL STATE ────────────────────────────────────
        if self.user_goal_centroid is not None:
            # --- Nudge goal to nearest walkable cell in C-space ---
            goal_cell = self._nearest_walkable(
                self._cspace_walkable, raw_goal_cell)
            if goal_cell is None:
                self.get_logger().error(
                    'No walkable cell near manual goal! Rejecting.')
                self._finish_manual_goal()
                return

            # Use the nudged world position for reach-distance checks
            nudged_goal_world = PathPlanner.grid_to_world(
                self.mapdata, goal_cell)

            if self._dist_to_point(nudged_goal_world) < self.GOAL_REACHED_DIST:
                self.get_logger().info('Manual waypoint reached!')
                self._finish_manual_goal()
                return

            path, _, _, _ = planner.a_star(start_cell, goal_cell)

            if path:
                self.user_goal_fails = 0
                self._publish_path(path, real_start, start_cell)
            else:
                self.user_goal_fails += 1
                if self.user_goal_fails <= 3:
                    self.get_logger().warning(
                        'Path to manual goal blocked '
                        f'({self.user_goal_fails}/3). Retrying...')
                else:
                    self.get_logger().error(
                        'Manual goal is unreachable! Cancelling command.')
                    self._finish_manual_goal()
            return
        # ──────────────────────────────────────────────────────────────

        # ── RETURNING HOME STATE ──────────────────────────────────────
        if self.returning_home:
            if self._dist_to_point(self.initial_pose) < self.GOAL_REACHED_DIST:
                self.get_logger().info(
                    'Safely returned to original position. '
                    'Exploration fully complete!')
                self.current_path = []
                self._publish_stop()
                self.is_finished_exploring = True
                self.exploration_complete = True

                # REVERT BACK TO NORMAL
                self.active_kernel = self.DEFAULT_KERNEL_SIZE
                self.return_home_fails = 0
                return

            raw_home = PathPlanner.world_to_grid(
                self.mapdata, self.initial_pose)

            home_cell = (
                self._nearest_walkable(self._cspace_walkable, raw_home)
                or raw_home)
            path, _, _, _ = planner.a_star(start_cell, home_cell)
            if (path and home_cell != raw_home
                    and self._dist_to_point(PathPlanner.grid_to_world(
                        self.mapdata, home_cell)) < self.GOAL_REACHED_DIST):
                # We reached the inflated boundary, not home. Let the existing
                # retry policy reduce inflation instead of republishing a
                # singleton path forever.
                path = None

            if path:
                self.return_home_fails = 0  # Reset fails, KEEP active_kernel
                self._publish_path(path, real_start, start_cell)
            else:
                self.return_home_fails += 1

                if self.return_home_fails <= 5:
                    self.active_kernel = max(
                        self._minimum_kernel, self.active_kernel - 2)
                    self.get_logger().warning(
                        'Path home blocked. Shrinking C-Space to '
                        f'{self.active_kernel} '
                        f'({self.return_home_fails}/5)...')
                else:
                    self.get_logger().error(
                        'Path completely blocked even with reduced '
                        'safety margin. Shutting down.')
                    self.current_path = []
                    self._publish_stop()
                    self.is_finished_exploring = True
                    self.exploration_complete = True
                    # Reset if we give up
                    self.active_kernel = self.DEFAULT_KERNEL_SIZE
            return
        # ──────────────────────────────────────────────────────────────

        # ── EXPLORATION STATE ─────────────────────────────────────────

        frontiers = self.search_frontiers(real_start)
        self._evaluate_frontiers(frontiers, planner, start_cell)

        lowest_cost = float('inf')
        best_path, best_centroid = self._resume_frontier_path(
            frontiers, planner, start_cell)

        # Filter the entire set before limiting search work. Previously five
        # large blocked/visited edges could permanently hide smaller openings.
        candidates = sorted(
            (f for f in frontiers if f.approach is not None),
            key=lambda f: self._dist_to_point(f.approach))
        for offset in range(0, len(candidates), self.PLAN_BATCH_SIZE):
            if best_path is not None:
                break
            batch = candidates[offset:offset + self.PLAN_BATCH_SIZE]
            goals = [PathPlanner.world_to_grid(self.mapdata, f.approach)
                     for f in batch]
            results = planner.a_star_many(start_cell, goals)
            for frontier, (path, distance, _, _) in zip(batch, results):
                if not path or distance is None:
                    continue
                cost = 10.0 * distance + 1.0 / frontier.size
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_path = path
                    best_centroid = frontier.approach

        if best_path and best_centroid:
            self.no_frontiers_found_counter = 0
            if best_centroid != self.current_goal_centroid:
                self.get_logger().info(
                    f'New frontier goal: ({best_centroid.x:.2f}, '
                    f'{best_centroid.y:.2f})')
            self.current_goal_centroid = best_centroid
            self.last_replan_time = self.get_clock().now()
            self.goal_start_time = self.get_clock().now()
            self.last_progress_time = time.monotonic()
            self.best_path_idx = 0

            self._publish_path(best_path, real_start, start_cell)
        else:
            self.current_path = []
            self._publish_stop()  # no valid path — stop Pure Pursuit
            # Retry temporary obstructions before declaring exhaustion.
            if (any(f.status == 'deferred' for f in frontiers)
                    and any(math.isfinite(item.expires)
                            for item in self._frontier_exclusions)):
                self.no_frontiers_found_counter = 0
                self._publish_markers(frontiers)
                return
            self.no_frontiers_found_counter += 1
            if self.no_frontiers_found_counter >= self.NUM_EXPLORE_FAILS:
                self.get_logger().info(
                    'No unvisited footprint-safe frontier viewpoints remain '
                    f'({len(frontiers)} boundary segments still visible). '
                    'Returning to original position...')
                self.exploration_complete = True
                self.returning_home = True
                self.current_goal_centroid = None
            else:
                self.get_logger().debug(
                    'Evaluating remaining frontiers. '
                    f'{self.no_frontiers_found_counter}/'
                    f'{self.NUM_EXPLORE_FAILS} fails before returning home.')
        self._publish_markers(frontiers)

    # ── marker visualisation ─────────────────────────────────────────

    def _refresh_markers(self):
        """Keep diagnostics visible during driving, pauses and mission completion."""
        if self.mapdata is None or self.pose is None:
            return
        start = PathPlanner.world_to_grid(self.mapdata, self.pose.position)
        frontiers = self.search_frontiers(start)
        self._evaluate_frontiers(frontiers, self._prepare_planning_grid(), start)
        self._publish_markers(frontiers)

    def _publish_markers(self, frontiers):
        ma = MarkerArray()
        frame = self.mapdata.header.frame_id or 'odom'
        stamp = self.get_clock().now().to_msg()

        def marker(namespace, index, kind):
            result = Marker()
            result.header.frame_id = frame
            result.header.stamp = stamp
            result.ns = namespace
            result.id = index
            result.type = kind
            result.action = Marker.ADD
            result.pose.orientation.w = 1.0
            ma.markers.append(result)
            return result

        clear = marker('frontiers', 0, Marker.SPHERE)
        clear.action = Marker.DELETEALL
        boundary = marker('frontier_boundaries', 0, Marker.POINTS)
        boundary.scale.x = boundary.scale.y = self.mapdata.info.resolution
        boundary.color.g = boundary.color.b = 1.0
        boundary.color.a = 0.85
        colors = {'reachable': (0.1, 1.0, 0.1),
                  'deferred': (1.0, 0.5, 0.0),
                  'blocked': (0.55, 0.55, 0.55)}
        for index, frontier in enumerate(frontiers):
            if frontier.cells is not None:
                boundary.points.extend(
                    PathPlanner.grid_to_world(self.mapdata, tuple(map(int, cell)))
                    for cell in frontier.cells)
            viewpoint = marker('frontier_viewpoints', index, Marker.SPHERE)
            viewpoint.pose.position = (frontier.approach if frontier.approach
                                       is not None else frontier.centroid)
            viewpoint.pose.position = Point(
                x=viewpoint.pose.position.x, y=viewpoint.pose.position.y, z=0.12)
            viewpoint.scale.x = viewpoint.scale.y = viewpoint.scale.z = 0.22
            viewpoint.color.r, viewpoint.color.g, viewpoint.color.b = colors[
                frontier.status]
            viewpoint.color.a = 0.95

        if self.current_goal_centroid is not None:
            active = marker('frontier_active', 0, Marker.SPHERE)
            goal = self.current_goal_centroid
            active.pose.position = Point(x=goal.x, y=goal.y, z=0.2)
            active.scale.x = active.scale.y = active.scale.z = 0.30
            active.color.r = active.color.g = active.color.a = 1.0
        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()