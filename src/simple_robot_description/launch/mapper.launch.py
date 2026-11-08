from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        parameters=[{
            'use_sim_time': True,
            'target_frame': 'lidar_link',
            # Collision-height returns, including a full view for safe turns.
            'min_height': 0.0,
            'max_height': 0.2,
            'angle_min': -3.141592653589793,
            'angle_max': 3.141592653589793,
            'range_min': 0.08,
            'range_max': 10.0,
            'angle_increment': 0.005,
            'scan_time': 0.1,
            'use_inf': True,
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
        parameters=[{'use_sim_time': True, 'pose_topic': '/lidar_odom'}],
        output='screen',
    )

    return LaunchDescription([
        pointcloud_to_laserscan,
        mapper,
    ])
