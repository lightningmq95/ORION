## Pose Estimation and Autonomous Mapping of unknown environment using multi-sensor fusion with Extended Kalman Filter

Tested on:

- Ubuntu 24.04 (ROS2 Jazzy) and Ubuntu 26.04 (ROS2 Lyrical).

Project uses:

- Gazebo for simulation.
- rviz2 for visualization.

### Install dependencies

Deactivate any active Conda environment so ROS uses the system Python:

```sh
conda deactivate
```

With your ROS environment sourced and `ROS_DISTRO` set, install Xacro, the
Gazebo simulator and ROS bridge, and point cloud to laserscan conversion:

```sh
sudo apt update
sudo apt install ros-$ROS_DISTRO-xacro
sudo apt install ros-$ROS_DISTRO-ros-gz-sim ros-$ROS_DISTRO-ros-gz-bridge
sudo apt install ros-$ROS_DISTRO-pointcloud-to-laserscan
```

### Build the env

```sh
colcon build
```

### Launch the Simulation

Source ROS and the workspace every time you open a new terminal. Repeat these
steps after installing ROS or Gazebo packages so their environment hooks add
the simulator executable, libraries, and plugins to the current shell.

If using bash run:

```sh
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
```

If using zsh run:

```sh
source /opt/ros/$ROS_DISTRO/setup.zsh
source install/setup.zsh
```

```sh
ros2 launch simple_robot_description gazebo.launch.py
```

You can specify a world with:

```sh
ros2 launch simple_robot_description gazebo.launch.py world_name:=empty/small_house/small_warehouse
```

### Start Extended Kalman Filter

```sh
ros2 launch simple_robot_description ekf.launch.py
```

The params of the Extended Kalman Filter can be tweaked in the ekf_params.yaml file inside simple_robot_description/config/ekf_params.yaml

### Start Vizualisation

```sh
python3 src/simple_robot_control/simple_robot_control/vizualize.py --record --output ./runs
```

### Start Mapper Node

```sh
ros2 launch simple_robot_description mapper.launch.py
```

### RViz2

From the project root, open RViz with the saved display configuration:

```sh
rviz2 -d config.rviz
```

The preset uses the `odom` fixed frame and enables the map (`/map`) and planned
path (`/pure_pursuit/path`) displays. Start the mapper and autonomous navigation
nodes to populate them. The display topics are already configured.

### Start Autonomous traversal Node with Frontier Exploration

```sh
ros2 launch simple_robot_description autonomous_nav.launch.py
```

Manual goals sent through RViz's **2D Goal Pose** tool interrupt the active
exploration task. After reaching the waypoint, the robot plans a fresh route
from its current position to the interrupted frontier if it is still valid
and reachable; otherwise it selects another remaining frontier. Replacing a
manual waypoint preserves the original interrupted task.

After all reachable frontiers have been discovered, a manual goal overrides
any return-home trip. The robot goes to the waypoint and stays there. Rejected
or unreachable manual goals also preserve this completed state; during active
exploration, those outcomes resume the interrupted task. If no eligible
frontiers remain when a manual command ends, the robot stays at its current
position.

### Run Robot Control Node (Manual Control using W,S,A,D (optional))

```sh
ros2 run simple_robot_control robot_subscriber
```

Run ros2 topic list to see the available topics from the simulation which you can use to access camera, odometry, lidar data, etc.

### If Gazebo shows errors and crashes, install it using

```sh
sudo pkill -f gazebo
sudo pkill -f ign
```
