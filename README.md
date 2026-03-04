
### Launch the Simulation

```sh
ros2 launch simple_robot_description gazebo.launch.py
```

You can specify a world with:
```sh
ros2 launch simple_robot_description gazebo.launch.py world_name:=simple_world/small_house/small_warehouse(any of them)
```

### Run Control Nodes in another terminal

```sh
ros2 run simple_robot_control robot_subscriber(W,S,A,D control).

#type-ros2 topic list (to find avaible topics from the simulation which you can use to access camera and other things.)
