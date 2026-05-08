#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from dataclasses import dataclass
from collections import deque
import math

from simple_robot_control.path_planner import PathPlanner


@dataclass
class Frontier:
    size: int
    centroid: Point


class FrontierExplorer(Node):

    # ── tuning ──────────────────────────────────────────────────────
    GOAL_REACHED_DIST   = 0.35    # metres; margin to mark a goal as reached (must be > grid resolution)
    REPLAN_COOLDOWN_S   = 2.0   # in seconds; Min cooldown time to look for new goals
    CURRENT_GOAL_BONUS  = 0.95  # A bonus given to the bot when it reaches a goal pose
    MIN_FRONTIER_SIZE   = 8     # Min number of contiguous edge cells to make a valid frontier
    NUM_EXPLORE_FAILS   = 15    # After max explore failures it concludes that the map is fully explored and returns home
    TOP_K_FRONTIERS     = 5     # detected 100 frontiers, it chooses top K to reduce CPU load
    GOAL_TIMEOUT_S      = 15.0  # seconds of NO PROGRESS before giving up (path-index based)
    DEFAULT_KERNEL_SIZE = 18    # Cspace thickness
    # ────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('frontier_explorer')

        self.path_pub   = self.create_publisher(Path,        '/pure_pursuit/path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/frontiers_vis',     10)

        self.create_subscription(Odometry,      '/odom_fused', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/map',        self.map_callback,  10)
        self.create_subscription(PoseStamped,   '/goal_pose',  self.goal_pose_callback, 10)

        self.pose       = None
        self.mapdata    = None

        # ── state management ────────────────────────────────────────
        self.initial_pose: Point | None = None           
        self.returning_home = False                      
        
        self.current_goal_centroid: Point | None = None  
        self.last_replan_time    = self.get_clock().now()
        self.goal_start_time     = self.get_clock().now()
        self.last_progress_time  = self.get_clock().now()   # reset when robot advances along path
        self.best_path_idx       = 0                         # furthest path cell index reached
        self.no_frontiers_found_counter = 0
        self.is_finished_exploring = False
        
        # --- Squeeze Mode State Variables ---
        self.return_home_fails = 0 
        self.user_goal_fails = 0
        self.active_kernel = self.DEFAULT_KERNEL_SIZE  # Holds the reduced kernel size while driving
        
        self.user_goal_centroid: Point | None = None
        self.pre_manual_pose: Point | None = None  # saved position before manual override
        
        self.blacklisted_centroids = []
        self.current_path = []  # grid cells of the active path (for obstacle checking)

        self.timer = self.create_timer(1.0, self.explore_loop)
        self.get_logger().info("Frontier Explorer started.")

    # ── callbacks ───────────────────────────────────────────────────

    def goal_pose_callback(self, msg: PoseStamped):
        self.user_goal_centroid = msg.pose.position
        
        # Save where the robot currently is so we can return after the manual goal
        if self.pose is not None:
            self.pre_manual_pose = Point(
                x=self.pose.position.x,
                y=self.pose.position.y,
                z=0.0
            )
        
        # Reset fails and kernel when a new goal is clicked
        self.user_goal_fails = 0
        self.active_kernel = self.DEFAULT_KERNEL_SIZE 
        
        self.get_logger().info(f"Manual Override: Navigating to ({self.user_goal_centroid.x:.2f}, {self.user_goal_centroid.y:.2f})")
        
        self.is_finished_exploring = False
        self.returning_home = False
        self.no_frontiers_found_counter = 0
        self.current_goal_centroid = None

    def odom_callback(self, msg: Odometry):
        self.pose = msg.pose.pose
        if self.initial_pose is None:
            self.initial_pose = Point()
            self.initial_pose.x = self.pose.position.x
            self.initial_pose.y = self.pose.position.y
            self.initial_pose.z = self.pose.position.z

    def map_callback(self, msg: OccupancyGrid):
        self.mapdata = msg

    # ── frontier detection ──────────────────────────────────────────

    def search_frontiers(self, start_cell):
        queue   = deque([start_cell])          
        visited = {start_cell}
        is_frontier: dict = {}
        frontiers = []

        while queue:
            current = queue.popleft()
            for neighbor in PathPlanner.neighbors_of_4(
                    self.mapdata, current, must_be_walkable=False):
                val = PathPlanner.get_cell_value(self.mapdata, neighbor)
                if val >= 0 and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                elif self._is_new_frontier_cell(neighbor, is_frontier):
                    is_frontier[neighbor] = True
                    frontier = self._build_frontier(neighbor, is_frontier)
                    if frontier.size >= self.MIN_FRONTIER_SIZE:
                        frontiers.append(frontier)
        return frontiers

    def _is_new_frontier_cell(self, cell, is_frontier):
        if PathPlanner.get_cell_value(self.mapdata, cell) != -1 or cell in is_frontier:
            return False
        for n in PathPlanner.neighbors_of_4(
                self.mapdata, cell, must_be_walkable=False):
            if 0 <= PathPlanner.get_cell_value(self.mapdata, n) < 50:
                return True
        return False

    def _build_frontier(self, initial_cell, is_frontier):
        frontier_cells = [initial_cell]
        is_frontier[initial_cell] = True
        queue = deque([initial_cell])

        while queue:
            current = queue.popleft()
            for neighbor in PathPlanner.neighbors_of_8(
                    self.mapdata, current, must_be_walkable=False):
                if self._is_new_frontier_cell(neighbor, is_frontier):
                    is_frontier[neighbor] = True
                    frontier_cells.append(neighbor)
                    queue.append(neighbor)

        size = len(frontier_cells)
        avg_x = sum(c[0] for c in frontier_cells) / size
        avg_y = sum(c[1] for c in frontier_cells) / size
        
        best_cell = min(frontier_cells, key=lambda c: (c[0]-avg_x)**2 + (c[1]-avg_y)**2)
        centroid = PathPlanner.grid_to_world(self.mapdata, best_cell)
        
        return Frontier(size=size, centroid=centroid)

    # ── helpers ─────────────────────────────────────────────────────

    def _dist_to_point(self, p: Point) -> float:
        if self.pose is None:
            return float('inf')
        dx = self.pose.position.x - p.x
        dy = self.pose.position.y - p.y
        return math.hypot(dx, dy)

    def _goal_still_valid(self, frontiers) -> bool:
        if self.current_goal_centroid is None:
            return False
        for f in frontiers:
            if math.hypot(f.centroid.x - self.current_goal_centroid.x,
                          f.centroid.y - self.current_goal_centroid.y) < 1.0:
                return True
        return False

    def _seconds_since_replan(self) -> float:
        dt = self.get_clock().now() - self.last_replan_time
        return dt.nanoseconds * 1e-9

    def _is_path_blocked(self) -> bool:
        """Check if the stored path ahead of the robot is now blocked by new obstacles."""
        if not self.current_path or self.mapdata is None or self.pose is None:
            return False

        robot_cell = PathPlanner.world_to_grid(self.mapdata, self.pose.position)

        # Find the nearest cell on the path to the robot
        min_dist_sq = float('inf')
        nearest_idx = 0
        for i, cell in enumerate(self.current_path):
            d_sq = (cell[0] - robot_cell[0]) ** 2 + (cell[1] - robot_cell[1]) ** 2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                nearest_idx = i

        # Only check path cells AHEAD of the robot (from nearest onward)
        for cell in self.current_path[nearest_idx:]:
            if not PathPlanner.is_cell_in_bounds(self.mapdata, cell):
                return True
            val = PathPlanner.get_cell_value(self.mapdata, cell)
            if val >= 50:  # only actual obstacles block the path (not unknown cells)
                return True
        return False

    def _is_at_end_of_path(self) -> bool:
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
        return dist < self.GOAL_REACHED_DIST

    def _get_nearest_path_idx(self) -> int:
        """Return the index of the path cell closest to the robot."""
        if not self.current_path or self.pose is None or self.mapdata is None:
            return 0
        robot_cell = PathPlanner.world_to_grid(self.mapdata, self.pose.position)
        min_dist_sq = float('inf')
        nearest_idx = 0
        for i, cell in enumerate(self.current_path):
            d_sq = (cell[0] - robot_cell[0]) ** 2 + (cell[1] - robot_cell[1]) ** 2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                nearest_idx = i
        return nearest_idx

    def _truncate_at_blind_turns(self, path):
        """Cut the path at the first significant heading change that borders
        unmapped (unknown) cells.  This forces the robot to approach unknown
        regions head-on so the LiDAR can map obstacles on all sides before
        the robot commits to a turn whose outcome is invisible.

        Returns the (possibly shortened) path.
        """
        HEADING_WINDOW  = 3             # cells to average heading over
        ANGLE_THRESHOLD = math.pi / 4   # 45° counts as a significant turn
        UNKNOWN_RADIUS  = 5             # cell radius to scan for unknown space
        MIN_KEEP        = 8             # always keep at least this many cells

        if len(path) < max(HEADING_WINDOW * 2 + 1, MIN_KEEP) or self.mapdata is None:
            return path

        for i in range(HEADING_WINDOW, len(path) - HEADING_WINDOW):
            # Smoothed heading before and after point i
            dx_in  = path[i][0] - path[i - HEADING_WINDOW][0]
            dy_in  = path[i][1] - path[i - HEADING_WINDOW][1]
            dx_out = path[i + HEADING_WINDOW][0] - path[i][0]
            dy_out = path[i + HEADING_WINDOW][1] - path[i][1]

            if (dx_in == 0 and dy_in == 0) or (dx_out == 0 and dy_out == 0):
                continue

            angle_in  = math.atan2(dy_in, dx_in)
            angle_out = math.atan2(dy_out, dx_out)
            turn = abs(angle_out - angle_in)
            if turn > math.pi:
                turn = 2.0 * math.pi - turn

            if turn < ANGLE_THRESHOLD:
                continue

            # Significant turn found — check for unknown cells nearby
            cx, cy = path[i]
            has_unknown = False
            for dx in range(-UNKNOWN_RADIUS, UNKNOWN_RADIUS + 1):
                for dy in range(-UNKNOWN_RADIUS, UNKNOWN_RADIUS + 1):
                    cell = (cx + dx, cy + dy)
                    if (PathPlanner.is_cell_in_bounds(self.mapdata, cell)
                            and PathPlanner.get_cell_value(self.mapdata, cell) == -1):
                        has_unknown = True
                        break
                if has_unknown:
                    break

            if has_unknown:
                # Keep a few cells before the turn so the robot has a target
                cut_idx = max(MIN_KEEP, i - 2)
                if cut_idx >= len(path):
                    return path  # nothing useful to truncate
                self.get_logger().info(
                    f"Blind turn detected at path[{i}] near unknown space — "
                    f"truncating {len(path)} → {cut_idx} cells "
                    f"(approaching straight to let LiDAR map first).")
                return path[:cut_idx]

        return path

    # ── main loop ───────────────────────────────────────────────────

    def explore_loop(self):
        if self.is_finished_exploring or self.pose is None or self.mapdata is None:
            return

        real_start = PathPlanner.world_to_grid(self.mapdata, self.pose.position)
        start_cell = real_start

        # --- Commit to current frontier goal ---
        # Check if we've reached/timed out the goal, or if path is blocked.
        # If goal is still valid and path is clear, skip expensive replan.
        if (self.user_goal_centroid is None
                and not self.returning_home
                and self.current_goal_centroid is not None):
            
            # Check: did we reach the goal?
            if self._dist_to_point(self.current_goal_centroid) < self.GOAL_REACHED_DIST:
                self.get_logger().info("Reached frontier goal — blacklisting to prevent loops.")
                self.blacklisted_centroids.append(self.current_goal_centroid)
                self.current_goal_centroid = None
                self.current_path = []
                self.path_pub.publish(Path())  # stop Pure Pursuit immediately
                # Fall through to pick a new goal
            
            # Check: is the robot making progress along the path?
            # Track which path cell the robot is nearest to; if the index
            # advances, the robot is following the path (even during U-turns
            # where Euclidean distance to the goal temporarily increases).
            else:
                current_idx = self._get_nearest_path_idx()
                if current_idx > self.best_path_idx + 2:  # advanced by >2 cells
                    self.best_path_idx = current_idx
                    self.last_progress_time = self.get_clock().now()

                stall_seconds = (self.get_clock().now() - self.last_progress_time).nanoseconds * 1e-9

                if stall_seconds > self.GOAL_TIMEOUT_S:
                    self.get_logger().warn(
                        f"No path progress for {stall_seconds:.1f}s "
                        f"(stuck at path cell {self.best_path_idx}/{len(self.current_path)}). "
                        f"Blacklisting and replanning.")
                    self.blacklisted_centroids.append(self.current_goal_centroid)
                    self.current_goal_centroid = None
                    self.current_path = []
                    self.best_path_idx = 0
                    self.path_pub.publish(Path())  # stop Pure Pursuit immediately
                    # Fall through to pick a new goal
            
                # Check: is the path blocked by a new obstacle?
                elif self._is_path_blocked():
                    self.get_logger().warn("Obstacle detected on path! Forcing immediate replan.")
                    self.current_goal_centroid = None
                    self.current_path = []
                    self.best_path_idx = 0
                    self.path_pub.publish(Path())  # stop Pure Pursuit immediately
                    # Fall through to replan
                
                # Check: have we completed a truncated path? (blind-turn safe approach)
                # The robot reached the end of the path but is still far from the
                # actual goal — replan now with the freshly-mapped surroundings.
                elif self._is_at_end_of_path():
                    self.get_logger().info(
                        "Reached end of truncated path — replanning with updated map.")
                    self.current_goal_centroid = None
                    self.current_path = []
                    self.best_path_idx = 0
                    self.path_pub.publish(Path())  # stop Pure Pursuit immediately
                    # Fall through to replan
                
                else:
                    # Goal is active, path is clear, making progress — keep going
                    return
        
        # Calculate C-Space using the locked active_kernel
        cspace   = PathPlanner.calc_cspace(self.mapdata, kernel_size=self.active_kernel)
        cost_map = PathPlanner.calc_cost_map(self.mapdata)

        # --- Escape the A* Start Trap ---
        # Nudge to a walkable cell for planning, but remember the real position
        if not PathPlanner.is_cell_walkable(cspace, start_cell):
            found_start = False
            for r in range(1, 15): 
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if abs(dx) == r or abs(dy) == r:
                            cand = (start_cell[0] + dx, start_cell[1] + dy)
                            if PathPlanner.is_cell_walkable(cspace, cand):
                                start_cell = cand
                                found_start = True
                                break
                        if found_start: break
                    if found_start: break
                if found_start: break

        # ── MANUAL USER GOAL STATE ────────────────────────────────────
        if self.user_goal_centroid is not None:
            # --- Safety Check: Reject goals placed directly inside known obstacles ---
            raw_goal_cell = PathPlanner.world_to_grid(self.mapdata, self.user_goal_centroid)
            if (PathPlanner.is_cell_in_bounds(self.mapdata, raw_goal_cell)
                    and PathPlanner.get_cell_value(self.mapdata, raw_goal_cell) >= 50):
                self.get_logger().error(
                    "Manual goal is inside a known obstacle! Rejecting.")
                self.current_path = []
                self.path_pub.publish(Path())
                self.user_goal_centroid = None
                self.active_kernel = self.DEFAULT_KERNEL_SIZE
                self.user_goal_fails = 0
                return

            # --- Nudge goal to nearest walkable cell in C-space ---
            goal_cell = raw_goal_cell
            if not PathPlanner.is_cell_walkable(cspace, goal_cell):
                found_goal = False
                for r in range(1, 15):
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            if abs(dx) == r or abs(dy) == r:
                                cand = (goal_cell[0] + dx, goal_cell[1] + dy)
                                if PathPlanner.is_cell_walkable(cspace, cand):
                                    goal_cell = cand
                                    found_goal = True
                                    break
                            if found_goal: break
                        if found_goal: break
                    if found_goal: break
                if not found_goal:
                    self.get_logger().error(
                        "No walkable cell near manual goal! Rejecting.")
                    self.current_path = []
                    self.path_pub.publish(Path())
                    self.user_goal_centroid = None
                    self.active_kernel = self.DEFAULT_KERNEL_SIZE
                    self.user_goal_fails = 0
                    return

            # Use the nudged world position for reach-distance checks
            nudged_goal_world = PathPlanner.grid_to_world(self.mapdata, goal_cell)

            if self._dist_to_point(nudged_goal_world) < self.GOAL_REACHED_DIST:
                self.get_logger().info("Manual waypoint reached!")
                self.current_path = []
                self.path_pub.publish(Path())
                self.user_goal_centroid = None
                self.active_kernel = self.DEFAULT_KERNEL_SIZE  
                self.user_goal_fails = 0
                
                # Check if frontiers remain to decide next action
                if self.pre_manual_pose is not None:
                    check_start = PathPlanner.world_to_grid(self.mapdata, self.pose.position)
                    frontiers = self.search_frontiers(check_start)
                    valid_frontiers = [
                        f for f in frontiers
                        if not any(
                            math.hypot(f.centroid.x - bc.x, f.centroid.y - bc.y) < 1.0
                            for bc in self.blacklisted_centroids
                        )
                    ]
                    
                    if valid_frontiers:
                        self.get_logger().info(
                            f"Frontiers remaining! Returning to pre-override position "
                            f"({self.pre_manual_pose.x:.2f}, {self.pre_manual_pose.y:.2f})")
                        self.user_goal_centroid = self.pre_manual_pose
                        self.pre_manual_pose = None
                        return
                    else:
                        self.get_logger().info("No frontiers remaining. Staying at current position.")
                        self.pre_manual_pose = None
                        self.is_finished_exploring = True
                        return
                
                return

            path, a_star_cost, _, _ = PathPlanner.a_star(cspace, cost_map, start_cell, goal_cell)

            if path:
                # Stitch robot's real position to the front if start was nudged
                if start_cell != real_start:
                    path = [real_start] + path
                path = self._truncate_at_blind_turns(path)
                self.user_goal_fails = 0
                self.current_path = path
                path_msg = PathPlanner.path_to_message(self.mapdata, path, self.get_clock().now().to_msg())
                self.path_pub.publish(path_msg)
            else:
                self.user_goal_fails += 1
                if self.user_goal_fails <= 3:
                    self.get_logger().warn(
                        f"Path to manual goal blocked ({self.user_goal_fails}/3). Retrying...")
                else:
                    self.get_logger().error("Manual goal is unreachable! Cancelling command.")
                    self.current_path = []
                    self.path_pub.publish(Path())
                    self.user_goal_centroid = None
                    self.active_kernel = self.DEFAULT_KERNEL_SIZE
                    self.user_goal_fails = 0
            return 
        # ──────────────────────────────────────────────────────────────

        # ── RETURNING HOME STATE ──────────────────────────────────────
        if self.returning_home:
            if self._dist_to_point(self.initial_pose) < self.GOAL_REACHED_DIST:
                self.get_logger().info("Safely returned to original position. Exploration fully complete!")
                self.current_path = []
                self.path_pub.publish(Path())
                self.is_finished_exploring = True
                
                # REVERT BACK TO NORMAL
                self.active_kernel = self.DEFAULT_KERNEL_SIZE
                self.return_home_fails = 0
                return

            home_cell = PathPlanner.world_to_grid(self.mapdata, self.initial_pose)
            
            if not PathPlanner.is_cell_walkable(cspace, home_cell):
                found_home = False
                for r in range(1, 15):
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            if abs(dx) == r or abs(dy) == r:
                                cand = (home_cell[0] + dx, home_cell[1] + dy)
                                if PathPlanner.is_cell_walkable(cspace, cand):
                                    home_cell = cand
                                    found_home = True
                                    break
                            if found_home: break
                        if found_home: break
                    if found_home: break

            path, a_star_cost, _, _ = PathPlanner.a_star(cspace, cost_map, start_cell, home_cell)

            if path:
                # Stitch robot's real position to the front if start was nudged
                if start_cell != real_start:
                    path = [real_start] + path
                path = self._truncate_at_blind_turns(path)
                self.return_home_fails = 0 # Reset fails, KEEP active_kernel
                self.current_path = path
                path_msg = PathPlanner.path_to_message(self.mapdata, path, self.get_clock().now().to_msg())
                self.path_pub.publish(path_msg)
            else:
                self.return_home_fails += 1
                
                if self.return_home_fails <= 5:
                    self.active_kernel = max(1, self.active_kernel - 2)
                    self.get_logger().warn(f"Path home blocked. Shrinking C-Space to {self.active_kernel} ({self.return_home_fails}/5)...")
                else:
                    self.get_logger().error("Path completely blocked even with reduced safety margin. Shutting down.")
                    self.current_path = []
                    self.path_pub.publish(Path())
                    self.is_finished_exploring = True
                    self.active_kernel = self.DEFAULT_KERNEL_SIZE # Reset if we give up
            return
        # ──────────────────────────────────────────────────────────────

        # ── EXPLORATION STATE ─────────────────────────────────────────
        
        # Ensure we always use the safest kernel for standard exploration
        self.active_kernel = self.DEFAULT_KERNEL_SIZE 
        
        frontiers  = self.search_frontiers(start_cell)
        self._publish_markers(frontiers)

        # If we still have a goal but its frontier disappeared, check path
        if not self._goal_still_valid(frontiers) and self.current_goal_centroid is not None:
            if self._is_path_blocked():
                self.get_logger().info("Current frontier gone AND path blocked — will replan.")
                self.current_goal_centroid = None
                self.current_path = []
            else:
                # Frontier centroid drifted but path is still clear — keep driving
                self.get_logger().debug("Frontier centroid drifted but path is clear. Staying committed.")
                return


        # We only reach here when current_goal_centroid is None (need a new goal)

        top_frontiers = sorted(frontiers, key=lambda f: f.size, reverse=True)[: self.TOP_K_FRONTIERS]

        lowest_cost = float('inf')
        best_path   = None
        best_centroid = None

        for f in top_frontiers:
            is_blacklisted = False
            for bc in self.blacklisted_centroids:
                if math.hypot(f.centroid.x - bc.x, f.centroid.y - bc.y) < 1.0:
                    is_blacklisted = True
                    break
            if is_blacklisted:
                continue

            goal_cell = PathPlanner.world_to_grid(self.mapdata, f.centroid)
            
            if not PathPlanner.is_cell_walkable(cspace, goal_cell):
                continue

            path, a_star_cost, _, _ = PathPlanner.a_star(cspace, cost_map, start_cell, goal_cell)
            # Stitch robot's real position to the front if start was nudged
            if path and start_cell != real_start:
                path = [real_start] + path

            if not path or not a_star_cost:
                continue

            cost = 10.0 * a_star_cost + (1.0 / f.size)

            if (self.current_goal_centroid is not None
                    and math.hypot(f.centroid.x - self.current_goal_centroid.x,
                                   f.centroid.y - self.current_goal_centroid.y) < 1.0):
                cost *= (1.0 - self.CURRENT_GOAL_BONUS)

            if cost < lowest_cost:
                lowest_cost   = cost
                best_path     = path
                best_centroid = f.centroid

        if best_path and best_centroid:
            self.no_frontiers_found_counter = 0  
            if best_centroid != self.current_goal_centroid:
                self.get_logger().info(f"New frontier goal: ({best_centroid.x:.2f}, {best_centroid.y:.2f})")
            self.current_goal_centroid = best_centroid
            self.last_replan_time      = self.get_clock().now()
            self.goal_start_time       = self.get_clock().now()
            self.last_progress_time    = self.get_clock().now()
            self.best_path_idx         = 0
            
            best_path = self._truncate_at_blind_turns(best_path)
            self.current_path = best_path
            path_msg = PathPlanner.path_to_message(
                self.mapdata, best_path, self.get_clock().now().to_msg())
            self.path_pub.publish(path_msg)
        else:
            self.no_frontiers_found_counter += 1
            self.current_path = []
            self.path_pub.publish(Path())  # no valid path — stop Pure Pursuit
            if self.no_frontiers_found_counter >= self.NUM_EXPLORE_FAILS:
                self.get_logger().info("All reachable frontiers discovered! Returning to original position...")
                self.returning_home = True
                self.current_goal_centroid = None
                self._publish_markers([])  
            else:
                self.get_logger().debug(f"Evaluating remaining frontiers. {self.no_frontiers_found_counter}/{self.NUM_EXPLORE_FAILS} fails before returning home.")

    # ── marker visualisation ─────────────────────────────────────────

    def _publish_markers(self, frontiers):
        ma = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        for i, bc in enumerate(self.blacklisted_centroids):
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns              = "blacklisted"
            m.id              = i + 1000  
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD
            m.pose.position   = bc
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.5
            ma.markers.append(m)

        for i, f in enumerate(frontiers[:self.TOP_K_FRONTIERS]):
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns              = "frontiers"
            m.id              = i + 1
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD
            m.pose.position   = f.centroid
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.2

            is_current = (
                self.current_goal_centroid is not None
                and math.hypot(
                    f.centroid.x - self.current_goal_centroid.x,
                    f.centroid.y - self.current_goal_centroid.y) < 1.0)

            m.color.r = 1.0 if is_current else 0.0
            m.color.g = 0.0 if is_current else 1.0
            m.color.b = 0.0
            m.color.a = 0.9
            ma.markers.append(m)

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


if __name__ == "__main__":
    main()