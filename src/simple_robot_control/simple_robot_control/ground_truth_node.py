# Authors: Akash Mohapatra, Manas Chintawar

#!/usr/bin/env python3
"""
Ground truth pose publisher that extracts robot pose from Gazebo's dynamic_pose/info
and republishes as PoseStamped with respect to world frame.
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped


class GroundTruthPublisher(Node):

    def __init__(self):
        super().__init__('ground_truth_node')

        # Declare parameters
        self.declare_parameter('robot_name', 'simple_robot')
        self.robot_name = self.get_parameter('robot_name').value

        # Subscribe to world poses
        self.pose_sub = self.create_subscription(
            TFMessage,
            '/world/default/dynamic_pose/info',
            self.pose_callback,
            10
        )

        # Publish ground truth pose
        self.ground_truth_pub = self.create_publisher(
            PoseStamped,
            '/ground_truth',
            10
        )

        self.get_logger().info(f'Ground truth publisher started for robot: {self.robot_name}')

    def pose_callback(self, msg: TFMessage):
        """Extract robot pose and republish."""
        # Find the transform for our robot
        for transform in msg.transforms:
            if transform.child_frame_id == self.robot_name:
                # Create PoseStamped message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = 'world'  # or 'odom' if you prefer
                
                # Position
                pose_msg.pose.position.x = transform.transform.translation.x
                pose_msg.pose.position.y = transform.transform.translation.y
                pose_msg.pose.position.z = transform.transform.translation.z
                
                # Orientation
                pose_msg.pose.orientation.x = transform.transform.rotation.x
                pose_msg.pose.orientation.y = transform.transform.rotation.y
                pose_msg.pose.orientation.z = transform.transform.rotation.z
                pose_msg.pose.orientation.w = transform.transform.rotation.w
                
                self.ground_truth_pub.publish(pose_msg)
                break


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()