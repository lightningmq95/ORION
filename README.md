## Pose Estimation and Mapping of unknown environment using multi-sensor fusion with Extended Kalman Filter

### Build the env

```sh
colcon build
```

### Launch the Simulation

Source the workspace everytime you open a new terminal.
If using bash use:

```sh
source install/setup.bash
```

If using zsh use:

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

### Start Extended Kalman Filter Node

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

### Run Robot Control Node

```sh
ros2 run simple_robot_control robot_subscriber(W,S,A,D control).

# Run ros2 topic list (to find available topics from the simulation which you can use to access camera and other things.)
```
