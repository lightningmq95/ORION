## Pose Estimation and Mapping of unknown environment using multi-sensor fusion with Extended Kalman Filter

### Build the env

```sh
colcon build
```

### Launch the Simulation

Source the workspace everytime you open a new terminal.

If using bash run:

```sh
source install/setup.bash
```

If using zsh run:

```sh
source install/setup.zsh
```

```sh
ros2 launch simple_robot_description gazebo.launch.py
```

You can specify a world with:

```sh
ros2 launch simple_robot_description gazebo.launch.py world_name:=simple_world/small_house/small_warehouse
```

### Start Extended Kalman Filter and Vizualisation Node

```sh
ros2 launch simple_robot_description ekf.launch.py
```

The params of the Extended Kalman Filter can be tweaked in the ekf_params.yaml file inside simple_robot_description/config/ekf_params.yaml

### Start Mapper Node

```sh
ros2 launch simple_robot_description mapper.launch.py
```

### RViz2

```sh
rviz2
```

Go to Add -> By Topic -> /map

### Start Autonomous traversal Node with Frontier Exploration

```sh
ros2 launch simple_robot_description autonomous_nav.launch.py
```

### Run Robot Control Node (Manual Control using W,S,A,D (optional))

```sh
ros2 run simple_robot_control robot_subscriber
```

Run ros2 topic list to see the available topics from the simulation which you can use to access camera, odometry, lidar data, etc.
