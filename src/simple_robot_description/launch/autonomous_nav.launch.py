from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pure_pursuit = Node(
        package='simple_robot_control',
        executable='pure_pursuit',
        name='pure_pursuit',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    frontier_explorer = Node(
        package='simple_robot_control',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([pure_pursuit, frontier_explorer])