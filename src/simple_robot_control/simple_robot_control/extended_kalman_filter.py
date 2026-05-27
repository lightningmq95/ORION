# Authors: Akash Mohapatra, Manas Chintawar

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import math
import numpy as np


class ExtendedKalmanFilterNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        self.state = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
        self.P = np.eye(3) * 1.0

        self.declare_parameter('process_noise_vx', 0.1)
        self.declare_parameter('process_noise_omega', 0.5)
        self.declare_parameter('imu_heading_noise', 0.05)

        process_vx    = self.get_parameter('process_noise_vx').value
        process_omega = self.get_parameter('process_noise_omega').value
        self.R_imu    = self.get_parameter('imu_heading_noise').value

        self.M = np.array([
            [process_vx, 0.0],
            [0.0, process_omega]
        ])

        self.last_odom_time = None

        self.create_subscription(Odometry, '/odom',        self.prediction_callback,   10)
        self.create_subscription(Imu,      '/imu',         self.imu_callback,          10)
        self.create_subscription(Odometry, '/lidar_odom',  self.lidar_odom_callback,   10)

        self.fused_odom_pub = self.create_publisher(Odometry, '/odom_fused', 10)

        self.get_logger().info('EKF Node started (Wheel Odometry + IMU + LiDAR Odometry).')

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    def normalize_angle(self, angle):
        while angle >  math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def prediction_callback(self, msg):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_odom_time is None:
            self.last_odom_time = current_time
            self.state[0] = msg.pose.pose.position.x
            self.state[1] = msg.pose.pose.position.y
            self.state[2] = self.quaternion_to_yaw(
                msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
            )
            return

        dt = current_time - self.last_odom_time
        self.last_odom_time = current_time

        if dt <= 0.0:
            return

        vx    = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        yaw_k = self.state[2]

        self.state[0] += vx * math.cos(yaw_k) * dt
        self.state[1] += vx * math.sin(yaw_k) * dt
        self.state[2]  = self.normalize_angle(self.state[2] + omega * dt)

        F = np.array([
            [1.0, 0.0, -vx * math.sin(yaw_k) * dt],
            [0.0, 1.0,  vx * math.cos(yaw_k) * dt],
            [0.0, 0.0,  1.0]
        ])

        V = np.array([
            [math.cos(yaw_k) * dt, 0.0],
            [math.sin(yaw_k) * dt, 0.0],
            [0.0,                  dt ]
        ])

        self.P = F @ self.P @ F.T + (V @ self.M @ V.T)

        self.publish_fused_odometry(msg.header.stamp)

    def imu_callback(self, msg):
        imu_yaw = self.quaternion_to_yaw(
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w
        )

        H = np.array([[0.0, 0.0, 1.0]])
        y_val = self.normalize_angle(imu_yaw - self.state[2])

        R = msg.orientation_covariance[8] if msg.orientation_covariance[8] > 0.0 else self.R_imu

        S = H @ self.P @ H.T + R
        K = self.P @ H.T / S[0, 0]

        self.state = self.state + K.flatten() * y_val
        self.state[2] = self.normalize_angle(self.state[2])
        IKH = np.eye(3) - K @ H
        self.P = IKH @ self.P @ IKH.T + K * R * K.T

    def lidar_odom_callback(self, msg):
        """UPDATE STEP: Fuse LiDAR odometry pose (x, y, yaw) into EKF."""
        lidar_yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        )

        z = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            lidar_yaw
        ])

        # Full state measurement
        H = np.eye(3)

        innovation = z - self.state
        innovation[2] = self.normalize_angle(innovation[2])

        # Read covariance from lidar_odom message (set in lidar_odom.py)
        R = np.diag([
            msg.pose.covariance[0],   # x noise
            msg.pose.covariance[7],   # y noise
            msg.pose.covariance[35]   # yaw noise
        ])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        self.state[2] = self.normalize_angle(self.state[2])
        IKH = np.eye(3) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ np.diag([R[0,0], R[1,1], R[2,2]]) @ K.T

    def publish_fused_odometry(self, timestamp):
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'

        odom.pose.pose.position.x = float(self.state[0])
        odom.pose.pose.position.y = float(self.state[1])
        odom.pose.pose.position.z = 0.0

        qx, qy, qz, qw = self.yaw_to_quaternion(self.state[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.pose.covariance[0]  = self.P[0, 0]
        odom.pose.covariance[7]  = self.P[1, 1]
        odom.pose.covariance[35] = self.P[2, 2]

        self.fused_odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = ExtendedKalmanFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()