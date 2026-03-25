#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
import message_filters
import math
import numpy as np
import cv2
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
        
        # Final output map
        self.grid_msg = OccupancyGrid()
        self.grid_msg.header.frame_id = 'odom'
        self.grid_msg.info.resolution = self.resolution
        self.grid_msg.info.width = self.width
        self.grid_msg.info.height = self.height
        self.grid_msg.info.origin.position.x = -(self.width * self.resolution) / 2.0
        self.grid_msg.info.origin.position.y = -(self.height * self.resolution) / 2.0
        self.grid_msg.info.origin.orientation.w = 1.0

        # Log-Odds tuning parameters 
        # Tighter bounds = faster response to dynamic changes
        self.L_OCC = 0.85          
        self.L_FREE = -0.4         
        self.MAX_LOG_ODDS = 3.5    # Reduced from 5.0 for sharper erasing of ghosts
        self.MIN_LOG_ODDS = -3.5  
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
        
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        
        self.publish_counter = 0
        self.publish_rate = 3 
        
        self.get_logger().info('Mapper Node started.')

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
        # ALSO: skip 1 out of every 2 rays for speed (subsampling). Detail is rarely lost.
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
        # We draw the free space as lines into a blank mask
        free_space_mask = np.zeros_like(self.log_odds, dtype=np.uint8)
        
        for hx, hy in zip(hit_x_grid, hit_y_grid):
            # Draw line from robot to hit point with thickness 1
            cv2.line(free_space_mask, (rx_grid, ry_grid), (hx, hy), 1, 1)

        # 4. Apply Updates to Map
        # Decrease log odds for free space (anywhere cv2.line drew a 1)
        self.log_odds[free_space_mask == 1] += self.L_FREE
        
        # Overwrite hits. Ensure we add L_OCC correctly by uniquely identifying hit cells
        self.log_odds[hit_y_grid, hit_x_grid] += (abs(self.L_FREE) + self.L_OCC) # Add back the L_FREE we just subtracted, plus L_OCC

        # 5. Fast Vectorized Clipping
        np.clip(self.log_odds, self.MIN_LOG_ODDS, self.MAX_LOG_ODDS, out=self.log_odds)

        # 6. Publish Map
        self.publish_counter += 1
        if self.publish_counter >= self.publish_rate:
            self.publish_counter = 0
            self.publish_map()

    def publish_map(self):
        # 1-step Vectorized thresholding
        final_map = np.where(self.log_odds > self.OCC_THRESHOLD, 100, 
                    np.where(self.log_odds < self.FREE_THRESHOLD, 0, -1)).astype(np.int8)
        
        self.grid_msg.header.stamp = self.get_clock().now().to_msg()
        self.grid_msg.data = final_map.ravel().tolist()
        
        self.map_pub.publish(self.grid_msg)

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