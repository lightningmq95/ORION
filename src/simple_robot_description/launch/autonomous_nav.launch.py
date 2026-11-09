"""Start the path planner using the running simulation, EKF and mapper."""

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():
    """Launch only the path planner node."""
    return LaunchDescription([
        Node(
            package='simple_robot_control',
            executable='path_planner',
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),
    ])
