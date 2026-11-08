import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('simple_robot_description')
    params_file = os.path.join(pkg_share, 'config', 'ekf_params.yaml')

    lidar_odom_node = Node(
        package='simple_robot_control',
        executable='lidar_odom',
        name='lidar_odom_node',
        output='screen',
        parameters=[
            {'min_points': 50},
            {'icp_max_dist': 0.5},
            {'cov_x': 0.01},
            {'cov_y': 0.01},
            {'cov_yaw': 0.03},
            # Gazebo already publishes odom -> base_footprint on /tf.
            {'publish_tf': False},
            {'use_sim_time': True},
        ],
    )

    ekf_node = Node(
        package='simple_robot_control',
        executable='ekf_node',
        name='ekf_node',
        output='screen',
        parameters=[params_file, {'use_sim_time': True, 'publish_tf': True}],
    )

    # vizualize = Node(
    #     package='simple_robot_control',
    #     executable='vizualize',
    #     name='odom_visualizer',
    #     arguments=['--record', '--output', './runs'],
    #     output='screen',
    # )
    return LaunchDescription([lidar_odom_node, ekf_node])
