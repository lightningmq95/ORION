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

The **Frontiers** display subscribes to `/frontiers_vis` and shows all currently
detected frontiers, including candidates the robot cannot presently reach:

- **Green:** connected candidate viewpoints.
- **Yellow:** the active goal.
- **Orange:** temporarily deferred frontiers.
- **Gray:** blocked frontiers.
- **Cyan:** frontier boundary cells.

Frontiers refresh while navigation runs. The retained marker snapshot also
appears when RViz opens after exploration has already started.

### Start Autonomous traversal Node with Frontier Exploration

```sh
ros2 launch simple_robot_description autonomous_nav.launch.py
```

Navigation gets collision dimensions from `/robot_description` and positions
all collision links through TF2. Both the planner and controller wait for the
complete model and use the same footprint and clearance margin. Changing the
URDF updates the navigation geometry; there is no fixed C-space kernel or
automatic reduction of safety clearance to force a route through an obstacle.

Live LiDAR obstacles are shared with the planner on
`/navigation/local_obstacles`. The controller evaluates forward steering arcs
and their braking clearance while the explorer checks the route ahead for
obstacles. A valid detour replaces the active route in one update. The robot
brakes when no safe motion is available. If it stays stuck and no frontier
route is available, it tries one short escape with fresh sensors. It prefers
forward motion, then backs out if the rear sweep is clear and retries planning.

Exploration finishes reachable frontier viewpoints in the current area before
moving to another area. Areas use a 4 m travel radius through known free space,
so frontiers across a wall are not treated as nearby. Within an area, selection
considers driving distance and turning time. Blocked viewpoints can be retried
later. The explorer tries alternate viewpoints and keeps the mission pending
while frontier boundaries remain without a safe route.

Adjust the area radius for your environment when launching, for example:

```sh
ros2 launch simple_robot_description autonomous_nav.launch.py local_area_radius:=5.0
```

Manual goals sent through RViz's **2D Goal Pose** tool interrupt the active
exploration task. After reaching the waypoint, the robot plans a fresh route
from its current position to the interrupted frontier if it is still valid
and reachable; otherwise it selects another remaining frontier. Replacing a
manual waypoint preserves the original interrupted task.

After all reachable frontiers have been discovered, a manual goal overrides
any return-home trip. The robot goes to the waypoint and stays there. Rejected
or unreachable manual goals also preserve this completed state; during active
exploration, those outcomes resume the interrupted task. If no reachable map
boundaries remain when a manual command ends, the robot stays at its current
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
