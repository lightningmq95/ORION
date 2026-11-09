from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    local_area_radius = DeclareLaunchArgument(
        'local_area_radius', default_value='4.0',
        description='Prefer local frontiers within this travel radius.')
    shared_parameters = {
        'use_sim_time': True,
        'safety_margin': 0.06,
    }
    pure_pursuit = Node(
        package='simple_robot_control',
        executable='pure_pursuit',
        name='pure_pursuit',
        output='screen',
        parameters=[shared_parameters],
    )

    frontier_explorer = Node(
        package='simple_robot_control',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=[shared_parameters, {
            'local_area_radius': ParameterValue(
                LaunchConfiguration('local_area_radius'), value_type=float),
        }],
    )

    stop_on_exit = [
        RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[EmitEvent(event=Shutdown(
                reason='Navigation exited; stopping the controller.'))]))
        for node in (pure_pursuit, frontier_explorer)
    ]
    return LaunchDescription(
        [local_area_radius] + stop_on_exit + [pure_pursuit, frontier_explorer])
