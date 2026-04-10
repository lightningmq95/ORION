import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('simple_robot_description')
    params_file = os.path.join(pkg_share, 'config', 'ekf_params.yaml')

    ekf_node = Node(
        package='simple_robot_control',
        executable='ekf_node',
        name='ekf_node',
        output='screen',
        parameters=[params_file, {'use_sim_time': True}],
    )

    vizualize = Node(
        package='simple_robot_control',
        executable='eval',
        name='odom_visualizer',
        arguments=['--record', '--output', './runs'],
        output='screen',
    )
    return LaunchDescription([ekf_node, vizualize])