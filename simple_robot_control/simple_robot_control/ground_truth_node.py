#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
import math

class GroundTruthNode(Node):
    def __init__(self):
        super().__init__('ground_truth_listener')
        
        # Subscribe to the bridged ground truth topic
        self.gt_sub = self.create_subscription(
            PoseArray,
            '/ground_truth',
            self.pose_callback,
            10
        )
        self.get_logger().info('Ground Truth Listener Node started.')

    def pose_callback(self, msg):
        if not msg.poses:
            return
            
        # The PoseArray contains poses of the model. 
        # Typically the first pose in the array is the main model pose.
        robot_pose = msg.poses[0]
        
        x = robot_pose.position.x
        y = robot_pose.position.y
        z = robot_pose.position.z
        
        # Extract quaternion
        qx = robot_pose.orientation.x
        qy = robot_pose.orientation.y
        qz = robot_pose.orientation.z
        qw = robot_pose.orientation.w
        
        # Simple yaw calculation from quaternion
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.get_logger().info(
            f'Ground Truth -> X: {x:.3f}, Y: {y:.3f}, Yaw: {math.degrees(yaw):.2f}°'
        )

def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()