#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

import cv2
import numpy as np


class OdomVisualizer(Node):

    def __init__(self):
        super().__init__('odom_visualizer')

        # subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Odometry, '/odom_fused', self.ekf_callback, 10)
        self.create_subscription(PoseStamped, '/ground_truth', self.gt_callback, 10)

        # trajectory buffers
        self.raw_traj = []
        self.ekf_traj = []
        self.gt_traj = []

        # RMSE variables
        self.error_sum = 0.0
        self.error_count = 0
        self.current_rmse = 0.0

        self.latest_ekf = None
        self.latest_gt = None

        # canvas
        self.size = 800

        # smooth scaling
        self.current_scale = 50.0
        self.current_cx = 0.0
        self.current_cy = 0.0

        self.timer = self.create_timer(0.05, self.draw)

        self.get_logger().info("ODOM VISUALIZER STARTED")


    # ----------- Dynamic scaling -----------
    def compute_target_view(self):
        all_points = self.raw_traj + self.ekf_traj + self.gt_traj

        if len(all_points) < 2:
            return self.current_scale, self.current_cx, self.current_cy

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        range_x = max_x - min_x
        range_y = max_y - min_y

        max_range = max(range_x, range_y)
        max_range = max(max_range, 1.0) * 1.2  # margin

        target_scale = self.size / max_range

        cx = (max_x + min_x) / 2.0
        cy = (max_y + min_y) / 2.0

        return target_scale, cx, cy


    # ----------- Smooth transition -----------
    def smooth_update(self, target_scale, target_cx, target_cy):

        # Fast zoom-out, smooth zoom-in
        if target_scale < self.current_scale:
            alpha = 0.3
        else:
            alpha = 0.08

        self.current_scale = (1 - alpha) * self.current_scale + alpha * target_scale
        self.current_cx = (1 - alpha) * self.current_cx + alpha * target_cx
        self.current_cy = (1 - alpha) * self.current_cy + alpha * target_cy


    # ----------- Coordinate transform -----------
    def world_to_pixel(self, x, y):
        # Shift to center
        x_shift = x - self.current_cx
        y_shift = y - self.current_cy

        # Rotate 90° anticlockwise: (x, y) → (-y, x)
        x_rot = -y_shift
        y_rot = x_shift

        # Convert to pixel
        px = int(self.size/2 + x_rot * self.current_scale)
        py = int(self.size/2 - y_rot * self.current_scale)

        return px, py


    # ----------- RMSE computation -----------
    def update_rmse(self):
        if self.latest_ekf is None or self.latest_gt is None:
            return

        ex, ey = self.latest_ekf
        gx, gy = self.latest_gt

        error = np.sqrt((ex - gx)**2 + (ey - gy)**2)

        self.error_sum += error**2
        self.error_count += 1

        self.current_rmse = np.sqrt(self.error_sum / self.error_count)


    # ----------- Callbacks -----------
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.raw_traj.append((x, y))


    def ekf_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.ekf_traj.append((x, y))

        self.latest_ekf = (x, y)
        self.update_rmse()


    def gt_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.gt_traj.append((x, y))

        self.latest_gt = (x, y)
        self.update_rmse()


    # ----------- Drawing -----------
    def draw(self):

        # update view
        target_scale, target_cx, target_cy = self.compute_target_view()
        self.smooth_update(target_scale, target_cx, target_cy)

        img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255

        # RAW ODOM (RED)
        for i in range(1, len(self.raw_traj)):
            p1 = self.world_to_pixel(*self.raw_traj[i-1])
            p2 = self.world_to_pixel(*self.raw_traj[i])
            cv2.line(img, p1, p2, (0,0,255), 2)

        # EKF ODOM (GREEN)
        for i in range(1, len(self.ekf_traj)):
            p1 = self.world_to_pixel(*self.ekf_traj[i-1])
            p2 = self.world_to_pixel(*self.ekf_traj[i])
            cv2.line(img, p1, p2, (0,200,0), 2)

        # GROUND TRUTH (BLUE)
        for i in range(1, len(self.gt_traj)):
            p1 = self.world_to_pixel(*self.gt_traj[i-1])
            p2 = self.world_to_pixel(*self.gt_traj[i])
            cv2.line(img, p1, p2, (255,0,0), 2)

        # legend
        cv2.putText(img, "RED -> RAW ODOM", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.putText(img, "GREEN -> EKF", (10,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

        cv2.putText(img, "BLUE -> GROUND TRUTH", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.putText(img, f"Scale: {self.current_scale:.2f} px/m", (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.putText(img, f"RMSE (EKF vs GT): {self.current_rmse:.3f} m", (10,130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow("ODOM TRAJECTORY", img)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = OdomVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()