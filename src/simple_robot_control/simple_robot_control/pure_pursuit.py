#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import Point, Twist, Vector3
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Import the utility from your package
from simple_robot_control.path_planner import PathPlanner

def euler_from_quaternion(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")

        self.cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_subscription(Odometry, "/odom_fused", self.update_odometry, 10)
        self.create_subscription(OccupancyGrid, "/map", self.update_map, 10)
        self.create_subscription(Path, "/pure_pursuit/path", self.update_path, 10)

        self.LOOKAHEAD_DISTANCE = 0.75  
        self.MAX_DRIVE_SPEED = 0.75     
        self.MAX_TURN_SPEED = 1.0       
        self.TURN_SPEED_KP = 1.0        
        self.DISTANCE_TOLERANCE = 0.25  

        self.OBSTACLE_AVOIDANCE_GAIN = 0.5
        self.FOV_DISTANCE = 20 
        self.FOV = 180 
        self.FOV_DEADZONE = 80 
        
        self.pose = None
        self.mapdata = None
        self.path = Path()
        self.reversed = False
        self.closest_distance = float("inf")
        
        # --- NEW: Hysteresis state variable ---
        self.is_spinning = False

        # --- Reactive LiDAR Safety Layer (bypasses map update latency) ---
        self.EMERGENCY_STOP_DISTANCE = 0.20  # meters — full stop
        self.SLOW_DOWN_DISTANCE = 0.50       # meters — begin speed reduction
        self.FRONT_ARC_DEGREES = 60          # ±60° from forward heading
        self.min_front_range = float("inf")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.create_subscription(LaserScan, '/lidar_2d_scan', self.update_lidar, sensor_qos)

        self.timer = self.create_timer(0.05, self.run_step)
        self.get_logger().info("Pure Pursuit Started.")

    def update_odometry(self, msg: Odometry):
        self.pose = msg.pose.pose

    def update_map(self, msg: OccupancyGrid):
        self.mapdata = msg

    def update_path(self, msg: Path):
        self.path = msg

    def update_lidar(self, msg: LaserScan):
        """Extract minimum range in the front arc for emergency collision avoidance."""
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        angles = (angles + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

        arc_rad = np.radians(self.FRONT_ARC_DEGREES)
        front_mask = np.abs(angles) <= arc_rad
        valid_mask = (ranges >= msg.range_min) & (ranges <= msg.range_max) & np.isfinite(ranges)

        combined = front_mask & valid_mask
        self.min_front_range = float(np.min(ranges[combined])) if np.any(combined) else float("inf")

    def distance(self, x0, y0, x1, y1):
        return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    def calculate_steering_adjustment(self):
        if not self.pose or not self.mapdata: return 0

        yaw = float(np.rad2deg(euler_from_quaternion(self.pose.orientation)))
        robot_cell = PathPlanner.world_to_grid(self.mapdata, self.pose.position)

        weighted_sum_of_angles = 0
        total_weight = 0
        self.closest_distance = float("inf")
        wall_cell_count = 0

        for dx in range(-self.FOV_DISTANCE, self.FOV_DISTANCE + 1):
            for dy in range(-self.FOV_DISTANCE, self.FOV_DISTANCE + 1):
                cell = (robot_cell[0] + dx, robot_cell[1] + dy)
                distance = PathPlanner.euclidean_distance(robot_cell, cell)

                if not PathPlanner.is_cell_in_bounds(self.mapdata, cell): continue
                is_wall = not PathPlanner.is_cell_walkable(self.mapdata, cell)
                
                if is_wall and distance < self.closest_distance:
                    self.closest_distance = distance

                angle = float(np.rad2deg(np.arctan2(dy, dx))) - yaw
                if self.reversed: angle += 180
                
                if angle < -180: angle += 360
                elif angle > 180: angle -= 360

                is_in_fov = (distance <= self.FOV_DISTANCE and -self.FOV/2 <= angle <= self.FOV/2 and not abs(angle) < self.FOV_DEADZONE/2)
                
                if not is_in_fov or not is_wall: continue

                weight = 1 / (distance**2) if distance != 0 else 0
                weighted_sum_of_angles += weight * angle
                total_weight += weight
                wall_cell_count += 1

        if total_weight == 0: return 0
        return -self.OBSTACLE_AVOIDANCE_GAIN * (weighted_sum_of_angles / total_weight) / wall_cell_count

    def send_speed(self, linear, angular):
        msg = Twist(linear=Vector3(x=float(linear)), angular=Vector3(z=float(angular)))
        self.cmd_vel.publish(msg)

    def stop(self):
        self.send_speed(0, 0)

    def run_step(self):
        if not self.pose or not self.path.poses:
            self.stop()
            return

        x = self.pose.position.x
        y = self.pose.position.y
        goal = self.path.poses[-1].pose.position

        if self.distance(x, y, goal.x, goal.y) < self.DISTANCE_TOLERANCE:
            self.stop()
            return

        nearest_idx = 0
        min_dist = float("inf")
        for i, p in enumerate(self.path.poses):
            d = self.distance(x, y, p.pose.position.x, p.pose.position.y)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        lookahead = goal
        for i in range(nearest_idx, len(self.path.poses)):
            p = self.path.poses[i]
            d = self.distance(x, y, p.pose.position.x, p.pose.position.y)
            if d > self.LOOKAHEAD_DISTANCE:
                lookahead = p.pose.position
                break

        yaw = euler_from_quaternion(self.pose.orientation)
        alpha = float(np.arctan2(lookahead.y - y, lookahead.x - x) - yaw)
        
        while alpha > np.pi: alpha -= 2 * np.pi
        while alpha < -np.pi: alpha += 2 * np.pi

        self.reversed = False
        
        # --- NEW: Hysteresis Logic ---
        # If we are far off target (> 60 deg), trigger spinning mode
        if abs(alpha) > (math.pi / 3):  
            self.is_spinning = True
        # If we are almost perfectly facing the target (< 10 deg), disable spinning mode
        elif abs(alpha) < 0.17:  
            self.is_spinning = False

        if self.is_spinning:
            # When spinning, DO NOT run obstacle avoidance. Just spin to the target.
            drive_speed = 0.0
            turn_speed = self.TURN_SPEED_KP * alpha
        else:
            # When driving forward, apply standard Pure Pursuit and Obstacle Avoidance
            drive_speed = self.MAX_DRIVE_SPEED
            lookahead_distance = self.distance(x, y, lookahead.x, lookahead.y)
            if np.sin(alpha) != 0:
                radius = lookahead_distance / (2 * np.sin(alpha))
                turn_speed = self.TURN_SPEED_KP * drive_speed / radius
            else:
                turn_speed = 0.0

            # Only add the obstacle avoidance repulsion force when driving forward
            turn_speed += self.calculate_steering_adjustment()
            
            # Slow down if approaching a wall (map-based, secondary layer)
            # closest_distance is in grid cells (1 cell = 0.05m resolution)
            if self.closest_distance < 10:  # ~0.5m from inflated wall boundary
                drive_speed *= 0.5 

        # --- LiDAR Emergency Reactive Layer ---
        # Directly uses raw LiDAR to prevent collisions with newly discovered
        # obstacles that haven't been mapped yet (bypasses map update latency).
        if drive_speed > 0:
            if self.min_front_range < self.EMERGENCY_STOP_DISTANCE:
                drive_speed = 0.0
            elif self.min_front_range < self.SLOW_DOWN_DISTANCE:
                speed_factor = (self.min_front_range - self.EMERGENCY_STOP_DISTANCE) / \
                               (self.SLOW_DOWN_DISTANCE - self.EMERGENCY_STOP_DISTANCE)
                drive_speed *= max(0.15, speed_factor)

        turn_speed = max(-self.MAX_TURN_SPEED, min(self.MAX_TURN_SPEED, turn_speed))
        self.send_speed(drive_speed, turn_speed)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()