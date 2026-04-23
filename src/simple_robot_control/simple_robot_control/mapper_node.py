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
        
        # Permanent obstacle memory: once a cell is confirmed occupied, it stays forever
        self.confirmed_obstacles = np.zeros((self.height, self.width), dtype=bool)
        self.CONFIRM_THRESHOLD = 2.0  # log-odds value to lock a cell as permanently occupied
        
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

        # ---------------------------------------------------------
        # --- 1. Obstacle Blobbing (Merging) Settings ---
        # ---------------------------------------------------------
        self.obstacle_merge_dist_m = 0.45  
        
        merge_kernel_cells = int(self.obstacle_merge_dist_m / self.resolution)
        if merge_kernel_cells % 2 == 0:  
            merge_kernel_cells += 1
            
        self.merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_kernel_cells, merge_kernel_cells))

        # ---------------------------------------------------------
        # --- 2. Obstacle Inflation (Safety Radius) Settings ---
        # ---------------------------------------------------------
        self.inflation_radius_m = 0.15 
        
        inflate_kernel_cells = int((self.inflation_radius_m / self.resolution) * 2) + 1
        if inflate_kernel_cells % 2 == 0:  
            inflate_kernel_cells += 1
            
        self.inflation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_kernel_cells, inflate_kernel_cells))

        # Log-Odds tuning parameters 
        self.L_OCC = 1.2           # Evidence per occupied hit (detect quickly)
        self.L_FREE = -0.15        # Evidence per free ray-through (erode slowly to preserve memory)
        self.MAX_LOG_ODDS = 5.0    # Higher cap = obstacles need many ray-throughs to erase
        self.MIN_LOG_ODDS = -2.0

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
            cv2.line(free_space_mask, (rx_grid, ry_grid), (hx, hy), 1, 1)

        # 4. Apply Updates to Map
        # Only apply free-space decay to cells that are NOT permanently confirmed obstacles
        free_update_cells = (free_space_mask == 1) & (~self.confirmed_obstacles)
        self.log_odds[free_update_cells] += self.L_FREE
        self.log_odds[hit_y_grid, hit_x_grid] += (abs(self.L_FREE) + self.L_OCC) 

        # 5. Fast Vectorized Clipping
        np.clip(self.log_odds, self.MIN_LOG_ODDS, self.MAX_LOG_ODDS, out=self.log_odds)
        
        # 6. Lock cells that have crossed the confirmed threshold
        self.confirmed_obstacles |= (self.log_odds >= self.CONFIRM_THRESHOLD)

        # 6. Publish Map
        self.publish_counter += 1
        if self.publish_counter >= self.publish_rate:
            self.publish_counter = 0
            self.publish_map()

    def publish_map(self):
        # Convert log-odds to probabilities (per your snippet)
        prob = 1.0 / (1.0 + np.exp(-self.log_odds))
        
        # 1. Create binary masks for OpenCV processing
        occupied_mask = (prob > 0.65).astype(np.uint8)
        free_mask = (prob < 0.35).astype(np.uint8)
        
        # 2. BLOBBING: Apply Morphological Closing
        # This dilates the obstacles to connect close points (chair legs), 
        # then erodes them so walls don't become massively thick.
        blobbed_occ = cv2.morphologyEx(occupied_mask, cv2.MORPH_CLOSE, self.merge_kernel)

        # 3. DILATION: Apply physical robot safety radius to the merged blocks
        inflated_occupied = cv2.dilate(blobbed_occ, self.inflation_kernel, iterations=1)
        
        # 4. Construct the RAW map
        raw_map = np.full(self.log_odds.shape, -1, dtype=np.int8)
        raw_map[free_mask == 1] = 0            # Lay down free space first
        raw_map[occupied_mask == 1] = 100      # Overwrite with raw obstacles
        
        # 5. Construct the INFLATED map
        inflated_map = np.full(self.log_odds.shape, -1, dtype=np.int8)
        inflated_map[free_mask == 1] = 0
        inflated_map[inflated_occupied == 1] = 100 # Overwrite with the fully processed blobs
        
        # 6. Add timestamps and publish BOTH maps
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