from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        parameters=[{
            'use_sim_time': True,
            'min_height': 0.05,
            'max_height': 0.1,
            'angle_min': -1.5708,
            'angle_max': 1.5708,
            'range_min': 0.08,
            'range_max': 10.0,
            'angle_increment': 0.0087,
            'scan_time': 0.1,
        }],
        remappings=[
            ('cloud_in', '/lidar/points'),
            ('scan', '/lidar_2d_scan'),
        ],
        output='screen',
    )

    mapper = Node(
        package='simple_robot_control',
        executable='mapper_node',
        name='simple_mapper',
        output='screen',
    )

    return LaunchDescription([
        pointcloud_to_laserscan,
        mapper,
    ])