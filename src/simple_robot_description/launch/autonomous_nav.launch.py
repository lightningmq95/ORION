from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

from launch_ros.actions import Node


def generate_launch_description():
    shared_geometry = {
        'use_sim_time': True,
        'robot_length': 0.46,
        'robot_width': 0.34,
        'safety_margin': 0.06,
    }
    pure_pursuit = Node(
        package='simple_robot_control',
        executable='pure_pursuit',
        name='pure_pursuit',
        output='screen',
        # The URDF wheel joints are 0.10 m ahead of base_footprint.
        parameters=[shared_geometry, {'drive_axle_offset_x': 0.10}],
    )

    frontier_explorer = Node(
        package='simple_robot_control',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=[shared_geometry],
    )

    stop_on_exit = [
        RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[EmitEvent(event=Shutdown(
                reason='Navigation exited; stopping the controller.'))]))
        for node in (pure_pursuit, frontier_explorer)
    ]
    return LaunchDescription(stop_on_exit + [pure_pursuit, frontier_explorer])
