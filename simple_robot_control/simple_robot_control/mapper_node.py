#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
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

        # Store latest odometry pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0  # From IMU (most reliable)
        self.odom_received = False
        self.imu_received = False

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/lidar', self.scan_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        
        # Publisher for the map
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        
        # Constants
        self.HITS_REQUIRED = 2  # Number of hits needed to mark as obstacle
        self.DECAY_INTERVAL = 100  # Decay map every N scans
        self.scan_count = 0
        
        self.get_logger().info('Simple Mapper Node started (IMU-based heading).')

    def odom_callback(self, msg):
        """Update position from odometry (more reliable for XY)"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.odom_received = True

    def imu_callback(self, msg):
        """Update heading from IMU (drift-free)"""
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        
        # Extract yaw from quaternion
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.imu_received = True

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
        if not self.odom_received or not self.imu_received:
            return

        self.scan_count += 1
        
        # Decay the map periodically to clear ghosts
        if self.scan_count % self.DECAY_INTERVAL == 0:
            self.decay_map()

        rx_grid, ry_grid = self.world_to_grid(self.robot_x, self.robot_y)
        if rx_grid is None:
            return

        for i, r in enumerate(msg.ranges):
            if r < msg.range_min or r > msg.range_max or math.isinf(r) or math.isnan(r):
                continue
                
            # Use IMU-based heading instead of odometry
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

    def decay_map(self):
        """Periodically clear old/unconfirmed data to reduce ghosting"""
        # Decay hit counts and convert back to uint8
        self.hit_count = (self.hit_count * 0.7).astype(np.uint8)
        
        # Set low-confidence cells back to unknown
        for y in range(self.height):
            for x in range(self.width):
                if self.hit_count[y, x] < self.HITS_REQUIRED:
                    idx = y * self.width + x
                    self.map_data[idx] = -1

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