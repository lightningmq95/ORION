import os
from os import pathsep
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
import xacro
from launch.actions import TimerAction


def generate_launch_description():
    
    # Package directories
    pkg_simple_robot = get_package_share_directory('simple_robot_description')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    
    # Model path setup for Gazebo
    model_path = str(Path(pkg_simple_robot).parent.resolve())
    model_path += pathsep + os.path.join(pkg_simple_robot)
    model_path += pathsep + os.path.join(pkg_simple_robot, 'models')
    gazebo_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH", model_path
    )
    
    # Paths
    urdf_file = os.path.join(pkg_simple_robot, 'urdf', 'simple_robot.urdf.xacro')
    world_file = os.path.join(pkg_simple_robot, 'config', 'simple_world.sdf')
    
    # Process xacro to get robot description
    robot_description_config = xacro.process_file(urdf_file)
    robot_description = {'robot_description': robot_description_config.toxml()}

    world_name_arg= DeclareLaunchArgument(name="world_name", default_value="empty")

    world_path = PathJoinSubstitution([
        pkg_simple_robot,
        "worlds",
        PythonExpression(expression=["'", LaunchConfiguration("world_name"), "'", " + '.world'"])
    ])
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', world_path]}.items(),
    )
    
    # Spawn robot (with delay)
    spawn_robot = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', 'simple_robot',
                    '-topic', 'robot_description',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '0.1',
                ],
                output='screen',
            )
        ]
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )
    
    # Bridge for LEFT camera (Gazebo->ROS uses [)
    bridge_camera_left = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/left/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
        ],
        output='screen'
    )
    
    # Bridge for RIGHT camera (Gazebo->ROS uses [)
    bridge_camera_right = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/right/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
        ],
        output='screen'
    )
    
    # Bridge for LEFT camera info (Gazebo->ROS uses [)
    bridge_camera_left_info = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/left/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        output='screen'
    )
    
    # Bridge for RIGHT camera info (Gazebo->ROS uses [)
    bridge_camera_right_info = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/right/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        output='screen'
    )
    
    # Bridge for 3D LiDAR point cloud (Gazebo->ROS uses [)
    bridge_lidar = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/lidar/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
        ],
        output='screen'
    )
    
    # Bridge for IMU data (Gazebo->ROS uses [)
    bridge_imu = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU',
        ],
        output='screen'
    )
    
    # Bridge for joint states (bidirectional @)
    bridge_joint_states = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/joint_states@sensor_msgs/msg/JointState@ignition.msgs.Model',
        ],
        output='screen'
    )
    
    # Bridge for odometry (Gazebo->ROS uses [)
    bridge_odom = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
        ],
        output='screen'
    )
    
    # Bridge for cmd_vel (ROS->Gazebo uses ])
    bridge_cmd_vel = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
        ],
        output='screen'
    )
    
    # Bridge for clock (Gazebo->ROS uses [)
    bridge_clock = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
        ],
        output='screen'
    )
    
    return LaunchDescription([
        gazebo_resource_path,
        world_name_arg,
        robot_state_publisher,
        gazebo,
        spawn_robot,
        bridge_camera_left,
        bridge_camera_right,
        bridge_camera_left_info,
        bridge_camera_right_info,
        bridge_lidar,
        bridge_imu,
        bridge_joint_states,
        bridge_odom,
        bridge_cmd_vel,
        bridge_clock,
    ])