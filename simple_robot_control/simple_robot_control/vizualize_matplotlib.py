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

        # canvas
        self.size = 800
        self.scale = 50.0   # pixels per meter

        self.timer = self.create_timer(0.05, self.draw)

        self.get_logger().info("ODOM VISUALIZER STARTED")


    def world_to_pixel(self, x, y):
        px = int(self.size/2 + x * self.scale)
        py = int(self.size/2 - y * self.scale)
        return px, py


    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.raw_traj.append((x, y))


    def ekf_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.ekf_traj.append((x, y))


    def gt_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.gt_traj.append((x, y))


    def draw(self):

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
        cv2.putText(img, "RED = RAW ODOM", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.putText(img, "GREEN = EKF", (10,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

        cv2.putText(img, "BLUE = GROUND TRUTH", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

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