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
         
        # State vector: [x, y, yaw] 
        self.state = np.array([0.0, 0.0, 0.0])
         
        # State covariance matrix (3x3) 
        self.P = np.eye(3) * 1.0
         
        # ===== PROCESS NOISE (Body frame velocities) =====  
        self.declare_parameter('process_noise_vx', 0.1)       
        self.declare_parameter('process_noise_vy', 0.01) # Wheels don't slip much sideways     
        self.declare_parameter('process_noise_yaw', 0.5)      
        self.declare_parameter('imu_heading_noise', 0.05)     

        # higher = less trust
        process_vx = self.get_parameter('process_noise_vx').value
        process_vy = self.get_parameter('process_noise_vy').value
        process_yaw = self.get_parameter('process_noise_yaw').value
        self.R_imu = self.get_parameter('imu_heading_noise').value
         
        # Variance of the motion model (Control Space Covariance M)
        self.M = np.array([ 
            [process_vx, 0.0, 0.0], 
            [0.0, process_vy, 0.0], 
            [0.0, 0.0, process_yaw] 
        ]) 

        # Timing 
        self.last_odom_time = None 
         
        # Subscriptions 
        self.create_subscription(Odometry, '/odom', self.prediction_callback, 10) 
        self.create_subscription(Imu, '/imu', self.imu_callback, 10) 
         
        # Publisher for fused odometry 
        self.fused_odom_pub = self.create_publisher(Odometry, '/odom_fused', 10) 
         
        self.get_logger().info('EKF Node started (Wheel Odometry + IMU Yaw).') 
     
    def quaternion_to_yaw(self, qx, qy, qz, qw): 
        siny_cosp = 2 * (qw * qz + qx * qy) 
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz) 
        return math.atan2(siny_cosp, cosy_cosp) 
     
    def yaw_to_quaternion(self, yaw): 
        qz = math.sin(yaw / 2.0) 
        qw = math.cos(yaw / 2.0) 
        return 0.0, 0.0, qz, qw 
     
    def normalize_angle(self, angle): 
        while angle > math.pi: angle -= 2 * math.pi 
        while angle < -math.pi: angle += 2 * math.pi 
        return angle 

    def prediction_callback(self, msg): 
        """ PREDICTION STEP: Driven by Wheel Odometry Velocities (Twist) """ 
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 
         
        if self.last_odom_time is None: 
            self.last_odom_time = current_time 
            # Initialize initial pose if available 
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

        # Extract local velocities from wheel odometry
        vx = msg.twist.twist.linear.x 
        vy = msg.twist.twist.linear.y 
        omega = msg.twist.twist.angular.z
         
        # Predict State using Runge-Kutta 2nd order (Midpoint method)
        yaw_k = self.state[2] 
        avg_yaw = yaw_k + (omega * dt / 2.0) 
         
        # Update the motion model to include vy
        delta_x = (vx * math.cos(avg_yaw) - vy * math.sin(avg_yaw)) * dt 
        delta_y = (vx * math.sin(avg_yaw) + vy * math.cos(avg_yaw)) * dt 
        delta_yaw = omega * dt 
         
        # Update State 
        self.state[0] += delta_x 
        self.state[1] += delta_y 
        self.state[2] = self.normalize_angle(self.state[2] + delta_yaw) 
         
        # Compute Jacobian of Motion Model w.r.t State (F) 
        F = np.array([ 
            [1.0, 0.0, (-vx * math.sin(avg_yaw) - vy * math.cos(avg_yaw)) * dt], 
            [0.0, 1.0,  (vx * math.cos(avg_yaw) - vy * math.sin(avg_yaw)) * dt], 
            [0.0, 0.0,  1.0] 
        ])

        # Compute Jacobian of Motion Model w.r.t Control inputs (V)
        V = np.array([
            [math.cos(avg_yaw) * dt, -math.sin(avg_yaw) * dt, 0.0],
            [math.sin(avg_yaw) * dt,  math.cos(avg_yaw) * dt, 0.0],
            [0.0,                    0.0,                   dt ]
        ])
         
        # Update Covariance (Project body-frame noise M through V) 
        self.P = F @ self.P @ F.T + (V @ self.M @ V.T)
         
        self.publish_fused_odometry(msg.header.stamp) 

    def imu_callback(self, msg): 
        """ UPDATE STEP: Driven by Absolute IMU Heading to lock in orientation """ 
        imu_yaw = self.quaternion_to_yaw( 
            msg.orientation.x, 
            msg.orientation.y, 
            msg.orientation.z, 
            msg.orientation.w 
        ) 
         
        # Measurement matrix (We only measure yaw directly) 
        H = np.array([[0.0, 0.0, 1.0]]) 
         
        # Innovation (Difference between true IMU and our predicted yaw) 
        y_val = self.normalize_angle(imu_yaw - self.state[2]) 
        y = np.array([y_val]) 
         
        # Observation Noise 
        if msg.orientation_covariance[8] > 0.0: 
            R = msg.orientation_covariance[8] 
        else: 
            R = self.R_imu 
             
        # Innovation Covariance 
        S = H @ self.P @ H.T + R 
         
        # Kalman Gain 
        K = self.P @ H.T / S[0, 0] 
         
        # Correct the State 
        self.state = self.state + K.flatten() * y_val 
        self.state[2] = self.normalize_angle(self.state[2]) 
         
        # Update Covariance 
        self.P = (np.eye(3) - K @ H) @ self.P 
     
    def publish_fused_odometry(self, timestamp): 
        odom = Odometry() 
        odom.header.stamp = timestamp 
        odom.header.frame_id = 'odom' 
        odom.child_frame_id = 'base_link' 
         
        # Pose 
        odom.pose.pose.position.x = float(self.state[0]) 
        odom.pose.pose.position.y = float(self.state[1]) 
        odom.pose.pose.position.z = 0.0 
         
        qx, qy, qz, qw = self.yaw_to_quaternion(self.state[2]) 
        odom.pose.pose.orientation.x = qx 
        odom.pose.pose.orientation.y = qy 
        odom.pose.pose.orientation.z = qz 
        odom.pose.pose.orientation.w = qw 
         
        # Populate Covariance 
        odom.pose.covariance[0] = self.P[0, 0] 
        odom.pose.covariance[7] = self.P[1, 1] 
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