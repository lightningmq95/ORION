#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
import math
import numpy as np

class SimpleMapperNode(Node):
    def __init__(self):
        super().__init__('simple_mapper')
        
        # Grid settings
        self.resolution = 0.05      # 5 cm per cell
        self.width = 400            # 20 meters wide
        self.height = 400           # 20 meters high
        
        # Initialize the map array with -1 (Unknown space)
        self.map_data = np.full(self.width * self.height, -1, dtype=np.int8)
        # Track hit counts per cell for occupancy confidence
        self.hit_count = np.zeros((self.width, self.height), dtype=np.uint8)

        # Store latest fused pose from EKF
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.fused_pose_received = False

        # Subscriptions
        # Subscribe to EKF fused odometry instead of raw sensors
        self.create_subscription(Odometry, '/odom_fused', self.fused_odom_callback, 10)
        self.create_subscription(LaserScan, '/lidar', self.scan_callback, 10)
        
        # Publisher for the map
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        
        # Constants
        self.HITS_REQUIRED = 2  # Number of hits needed to mark as obstacle
        self.DECAY_INTERVAL = 100  # Decay map every N scans
        self.scan_count = 0
        
        self.get_logger().info('Simple Mapper Node started (using EKF fused odometry).')

    def fused_odom_callback(self, msg):
        """Update pose from EKF fused odometry"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.fused_pose_received = True

    def world_to_grid(self, x, y):
        # Offset by half width/height so origin is center
        gx = int((x / self.resolution) + (self.width / 2))
        gy = int((y / self.resolution) + (self.height / 2))
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return gx, gy
        return None, None

    def bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return points

    def scan_callback(self, msg):
        # Wait for EKF fused pose
        if not self.fused_pose_received:
            return

        self.scan_count += 1

        rx_grid, ry_grid = self.world_to_grid(self.robot_x, self.robot_y)
        if rx_grid is None:
            return

        for i, r in enumerate(msg.ranges):
            if r < msg.range_min or r > msg.range_max or math.isinf(r) or math.isnan(r):
                continue
                
            # Use EKF-fused heading with laser scan
            beam_angle = self.robot_yaw + msg.angle_min + (i * msg.angle_increment)
            hit_x = self.robot_x + r * math.cos(beam_angle)
            hit_y = self.robot_y + r * math.sin(beam_angle)
            
            hx_grid, hy_grid = self.world_to_grid(hit_x, hit_y)
            if hx_grid is None:
                continue

            # Mark free space
            line_points = self.bresenham_line(rx_grid, ry_grid, hx_grid, hy_grid)
            for (px, py) in line_points[:-1]:
                idx = py * self.width + px
                self.map_data[idx] = 0

            # Mark obstacles with confidence (multiple hits required)
            self.hit_count[hy_grid, hx_grid] += 1
            if self.hit_count[hy_grid, hx_grid] >= self.HITS_REQUIRED:
                idx = hy_grid * self.width + hx_grid
                self.map_data[idx] = 100

        self.publish_map()

    def publish_map(self):
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'odom'
        
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        
        grid.info.origin.position.x = -(self.width * self.resolution) / 2.0
        grid.info.origin.position.y = -(self.height * self.resolution) / 2.0
        grid.info.origin.orientation.w = 1.0
        
        grid.data = self.map_data.tolist()
        self.map_pub.publish(grid)

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