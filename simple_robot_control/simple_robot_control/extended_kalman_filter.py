#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
import math
import numpy as np

class ExtendedKalmanFilterNode(Node):
    def __init__(self):
        super().__init__('ekf_node')
        
        # State vector: [x, y, yaw]
        self.state = np.array([0.0, 0.0, 0.0])
        
        # State covariance matrix (3x3)
        self.P = np.eye(3) * 0.1
        
        # ===== PROCESS NOISE (How much we trust motion model) =====
        # Tunable parameters - adjust based on simulation/real-world performance
        self.declare_parameter('process_noise_xy', 0.01)      # x,y position drift per update
        self.declare_parameter('process_noise_yaw', 0.05)     # yaw drift per update
        
        # ===== MEASUREMENT NOISE (Sensor uncertainty) =====
        # Odometry noise (typical wheeled robot)
        self.declare_parameter('odom_noise_xy', 0.1)          # odometry xy uncertainty
        self.declare_parameter('odom_noise_yaw', 0.2)         # odometry yaw uncertainty
        
        # IMU heading noise (typical MEMS IMU)
        self.declare_parameter('imu_heading_noise', 0.05)     # IMU yaw uncertainty
        
        # Get parameters
        process_xy = self.get_parameter('process_noise_xy').value
        process_yaw = self.get_parameter('process_noise_yaw').value
        odom_xy = self.get_parameter('odom_noise_xy').value
        odom_yaw = self.get_parameter('odom_noise_yaw').value
        imu_heading = self.get_parameter('imu_heading_noise').value
        
        self.Q = np.array([
            [process_xy, 0.0, 0.0],
            [0.0, process_xy, 0.0],
            [0.0, 0.0, process_yaw]
        ])
        
        self.R_odom = np.array([
            [odom_xy, 0.0, 0.0],
            [0.0, odom_xy, 0.0],
            [0.0, 0.0, odom_yaw]
        ])
        
        self.R_imu = imu_heading
        
        # Previous odometry for delta calculation
        self.prev_odom_x = 0.0
        self.prev_odom_y = 0.0
        self.prev_odom_yaw = 0.0
        self.prev_odom_set = False
        
        # Subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        
        # Publisher for fused odometry
        self.fused_odom_pub = self.create_publisher(Odometry, '/odom_fused', 10)
        
        self.get_logger().info('Extended Kalman Filter Node started.')
        self.get_logger().info(f'Process noise (xy): {process_xy}, (yaw): {process_yaw}')
        self.get_logger().info(f'Odometry noise (xy): {odom_xy}, (yaw): {odom_yaw}')
        self.get_logger().info(f'IMU heading noise: {imu_heading}')
    
    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        qx = 0.0
        qy = 0.0
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qx, qy, qz, qw
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def motion_model(self, delta_x, delta_y, delta_yaw):
        """
        Predict state based on odometry delta.
        Transform odometry delta from robot frame to world frame.
        """
        # Rotate odometry delta to world frame using current yaw
        avg_yaw = self.state[2] + delta_yaw / 2.0
        
        # Odometry gives local (robot frame) motion, convert to world frame
        world_delta_x = delta_x * math.cos(avg_yaw) - delta_y * math.sin(avg_yaw)
        world_delta_y = delta_x * math.sin(avg_yaw) + delta_y * math.cos(avg_yaw)
        
        # Update state
        self.state[0] += world_delta_x
        self.state[1] += world_delta_y
        self.state[2] = self.normalize_angle(self.state[2] + delta_yaw)
    
    def jacobian_motion(self, delta_x, delta_y, delta_yaw):
        """
        Compute Jacobian of motion model.
        F = dh/dx (derivative of state transition with respect to state)
        """
        avg_yaw = self.state[2] + delta_yaw / 2.0
        
        F = np.array([
            [1.0, 0.0, -delta_x * math.sin(avg_yaw) - delta_y * math.cos(avg_yaw)],
            [0.0, 1.0,  delta_x * math.cos(avg_yaw) - delta_y * math.sin(avg_yaw)],
            [0.0, 0.0, 1.0]
        ])
        return F
    
    def odom_callback(self, msg):
        """Process odometry measurements"""
        # On first message, just initialize
        if not self.prev_odom_set:
            self.prev_odom_x = msg.pose.pose.position.x
            self.prev_odom_y = msg.pose.pose.position.y
            self.prev_odom_yaw = self.quaternion_to_yaw(
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            )
            self.prev_odom_set = True
            return
        
        # Extract current pose from odometry
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        odom_yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        
        # Calculate delta motion
        delta_x = odom_x - self.prev_odom_x
        delta_y = odom_y - self.prev_odom_y
        delta_yaw = self.normalize_angle(odom_yaw - self.prev_odom_yaw)
        
        # Store current odometry for next iteration
        self.prev_odom_x = odom_x
        self.prev_odom_y = odom_y
        self.prev_odom_yaw = odom_yaw
        
        # ===== PREDICTION (Motion Model) =====
        # Compute Jacobian
        F = self.jacobian_motion(delta_x, delta_y, delta_yaw)
        
        # Predict state
        self.motion_model(delta_x, delta_y, delta_yaw)
        
        # Update covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        
        # ===== UPDATE (Measurement Model - Odometry) =====
        # Measurement residual (difference between measurement and predicted state)
        z = np.array([odom_x, odom_y, odom_yaw])
        y = z - self.state  # innovation
        y[2] = self.normalize_angle(y[2])
        
        # Measurement matrix (we measure x, y, yaw directly)
        H = np.eye(3)
        
        # Extract measurement covariance from message, use defaults if all-zero
        R = self.extract_odom_covariance(msg)
        
        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + R
        
        # Kalman gain: K = P * H^T * S^-1
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("Singular matrix in Kalman gain calculation")
            K = self.P @ H.T / (np.diag(S) + 1e-6)
        
        # Update state: x = x + K * y
        self.state = self.state + K @ y
        self.state[2] = self.normalize_angle(self.state[2])
        
        # Update covariance: P = (I - K * H) * P
        self.P = (np.eye(3) - K @ H) @ self.P
        
        self.publish_fused_odometry(msg.header.stamp)
    
    def extract_odom_covariance(self, msg):
        """
        Extract odometry covariance from message.
        If all zeros (simulated), use parameter defaults.
        """
        # Odometry covariance is 6x6: [x, y, z, rx, ry, rz]
        # We use indices: [0]=x, [7]=y, [35]=rz (yaw)
        cov_array = np.array(msg.pose.covariance)
        
        x_cov = cov_array[0]
        y_cov = cov_array[7]
        yaw_cov = cov_array[35]
        
        # If all zeros, use defaults
        if x_cov == 0.0 and y_cov == 0.0 and yaw_cov == 0.0:
            return self.R_odom
        
        # Otherwise use message values (clamp to reasonable ranges)
        x_cov = max(x_cov, 1e-6)
        y_cov = max(y_cov, 1e-6)
        yaw_cov = max(yaw_cov, 1e-6)
        
        return np.array([
            [x_cov, 0.0, 0.0],
            [0.0, y_cov, 0.0],
            [0.0, 0.0, yaw_cov]
        ])
    
    def imu_callback(self, msg):
        """Process IMU heading measurement"""
        imu_yaw = self.quaternion_to_yaw(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )
        
        # ===== UPDATE (Measurement Model - IMU Heading Only) =====
        # Only use yaw (index 2)
        H = np.array([[0.0, 0.0, 1.0]])  # Only measure yaw
        
        # Innovation (heading error)
        y = np.array([self.normalize_angle(imu_yaw - self.state[2])])
        
        # Extract IMU yaw covariance, use default if all-zero
        if msg.orientation_covariance[8] == 0.0:  # cov[8] is yaw variance
            R = self.R_imu
        else:
            R = max(msg.orientation_covariance[8], 1e-6)
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        if S[0, 0] > 1e-6:
            K = self.P @ H.T / S[0, 0]
        else:
            K = self.P @ H.T / 1e-6
        
        # Update state (only yaw)
        delta_yaw = K[2, 0] * y[0]
        self.state[2] = self.normalize_angle(self.state[2] + delta_yaw)
        
        # Update covariance
        self.P = (np.eye(3) - K @ H) @ self.P
    
    def publish_fused_odometry(self, timestamp):
        """Publish fused odometry"""
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Position
        odom.pose.pose.position.x = self.state[0]
        odom.pose.pose.position.y = self.state[1]
        odom.pose.pose.position.z = 0.0
        
        # Orientation (from yaw)
        qx, qy, qz, qw = self.yaw_to_quaternion(self.state[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        
        # Pose covariance (36-element ROS standard)
        odom.pose.covariance[0] = self.P[0, 0]   # x variance
        odom.pose.covariance[7] = self.P[1, 1]   # y variance
        odom.pose.covariance[35] = self.P[2, 2]  # yaw variance
        
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