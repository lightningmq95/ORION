from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    manual_control = Node(
        package='simple_robot_control',
        executable='robot_subscriber',
        name='robot_keyboard_control',
        output='screen',
    )

    return LaunchDescription([
        manual_control,
    ])