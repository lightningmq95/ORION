#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
import message_filters
import math
import numpy as np
import cv2
import copy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class SimpleMapperNode(Node):
    def __init__(self):
        super().__init__('simple_mapper')
        
        # Grid settings
        self.resolution = 0.05      # 5 cm per cell
        self.width = 600            # 30 meters wide
        self.height = 600           # 30 meters high
        
        # Bayesian Log-Odds Array
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Base setup for the Raw Map message
        self.grid_msg = OccupancyGrid()
        self.grid_msg.header.frame_id = 'odom'
        self.grid_msg.info.resolution = self.resolution
        self.grid_msg.info.width = self.width
        self.grid_msg.info.height = self.height
        self.grid_msg.info.origin.position.x = -(self.width * self.resolution) / 2.0
        self.grid_msg.info.origin.position.y = -(self.height * self.resolution) / 2.0
        self.grid_msg.info.origin.orientation.w = 1.0

        # Create an exact copy of the metadata for the Inflated Map message
        self.inflated_grid_msg = copy.deepcopy(self.grid_msg)

        # --- Obstacle Inflation (Blobbing) Settings ---
        self.inflation_radius_m = 0.15 # Inflate obstacles by 5 cm (adjust as needed)
        
        # Calculate kernel size based on resolution. Must be an odd number.
        kernel_size = int((self.inflation_radius_m / self.resolution) * 2) + 1
        if kernel_size % 2 == 0:  # Failsafe to ensure it's odd
            kernel_size += 1
            
        # Create a circular kernel for uniform expansion
        self.inflation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Log-Odds tuning parameters 
        self.L_OCC = 0.85          
        self.L_FREE = -0.4         
        self.MAX_LOG_ODDS = 3.5
        self.MIN_LOG_ODDS = -2.0
        self.OCC_THRESHOLD = 0.8   
        self.FREE_THRESHOLD = -0.3 

        self.MAX_TRUSTED_RANGE = 10.0 # Truncate long-distance blur

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom_fused')
        self.lidar_sub = message_filters.Subscriber(self, LaserScan, '/lidar_2d_scan', qos_profile=sensor_qos)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub], queue_size=20, slop=0.02
        )
        self.ts.registerCallback(self.sync_callback)
        
        # --- Create our TWO publishers ---
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, '/map_inflated', 1)
        
        self.publish_counter = 0
        self.publish_rate = 3 
        
        self.get_logger().info('Mapper Node started publishing to /map and /map_inflated.')

    def sync_callback(self, odom_msg, scan_msg):
        # 1. Exact pose extraction
        rx = odom_msg.pose.pose.position.x
        ry = odom_msg.pose.pose.position.y
        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w
        
        ryaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

        # Core robot location in grid
        rx_grid = int((rx / self.resolution) + (self.width / 2))
        ry_grid = int((ry / self.resolution) + (self.height / 2))
        
        if not (0 <= rx_grid < self.width and 0 <= ry_grid < self.height):
            return

        # 2. Fully Vectorized Lidar Math
        ranges = np.array(scan_msg.ranges)
        
        # Valid mask: within min/max, not nan, not infinite.
        valid_mask = (ranges >= scan_msg.range_min) & (ranges <= min(scan_msg.range_max, self.MAX_TRUSTED_RANGE)) & np.isfinite(ranges)
        # Apply subsampling mask (e.g., take every 2nd valid point) to halve raycasting work
        valid_mask_idx = np.where(valid_mask)[0][::2] 
        
        valid_ranges = ranges[valid_mask_idx]
        valid_angles = ryaw + scan_msg.angle_min + (valid_mask_idx * scan_msg.angle_increment)

        # Vectorized translation to map frame
        hit_x_grid = np.round((rx + valid_ranges * np.cos(valid_angles)) / self.resolution + (self.width / 2)).astype(np.int32)
        hit_y_grid = np.round((ry + valid_ranges * np.sin(valid_angles)) / self.resolution + (self.height / 2)).astype(np.int32)

        # Validate hit points bounds
        in_bounds = (hit_x_grid >= 0) & (hit_x_grid < self.width) & (hit_y_grid >= 0) & (hit_y_grid < self.height)
        hit_x_grid = hit_x_grid[in_bounds]
        hit_y_grid = hit_y_grid[in_bounds]

        # 3. OpenCV C++ Raytracing (INSANELY fast compared to Python Bresenham)
        free_space_mask = np.zeros_like(self.log_odds, dtype=np.uint8)
        
        for hx, hy in zip(hit_x_grid, hit_y_grid):
            # Draw line from robot to hit point with thickness 1
            cv2.line(free_space_mask, (rx_grid, ry_grid), (hx, hy), 1, 1)

        # 4. Apply Updates to Map
        # Decrease log odds for free space (anywhere cv2.line drew a 1)
        self.log_odds[free_space_mask == 1] += self.L_FREE
        
        # Overwrite hits. Ensure we add L_OCC correctly by uniquely identifying hit cells
        self.log_odds[hit_y_grid, hit_x_grid] += (abs(self.L_FREE) + self.L_OCC) 

        # 5. Fast Vectorized Clipping
        np.clip(self.log_odds, self.MIN_LOG_ODDS, self.MAX_LOG_ODDS, out=self.log_odds)

        # 6. Publish Map
        self.publish_counter += 1
        if self.publish_counter >= self.publish_rate:
            self.publish_counter = 0
            self.publish_map()

    def publish_map(self):
        # 1. Extract raw occupied and free masks based on thresholds
        occupied_mask = (self.log_odds > self.OCC_THRESHOLD).astype(np.uint8)
        free_mask = (self.log_odds < self.FREE_THRESHOLD).astype(np.uint8)
        
        # 2. DILATION: Grow the obstacles to merge close ones into blobs
        inflated_occupied = cv2.dilate(occupied_mask, self.inflation_kernel, iterations=1)
        
        # 3. Construct the RAW map
        raw_map = np.full(self.log_odds.shape, -1, dtype=np.int8)
        raw_map[free_mask == 1] = 0
        raw_map[occupied_mask == 1] = 100
        
        # 4. Construct the INFLATED map
        inflated_map = np.full(self.log_odds.shape, -1, dtype=np.int8)
        inflated_map[free_mask == 1] = 0
        inflated_map[inflated_occupied == 1] = 100 # Inflated areas override free space
        
        # 5. Add timestamps and publish BOTH maps
        current_time = self.get_clock().now().to_msg()
        
        # Publish Raw
        self.grid_msg.header.stamp = current_time
        self.grid_msg.data = raw_map.ravel().tolist()
        self.map_pub.publish(self.grid_msg)
        
        # Publish Inflated
        self.inflated_grid_msg.header.stamp = current_time
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