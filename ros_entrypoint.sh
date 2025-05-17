#!/bin/bash
set -e

# setup ros environment
source "/root/catkin_ws/devel/setup.bash"

# Setting environment variable for ROS
ROS_IP=$(ip addr show eth0 | grep -Po 'inet \K[\d.]+')
export ROS_IP=$ROS_IP

# eval  "roslaunch franka_gazebo franka_world.launch"
# eval "roslaunch franka_moveit_experiments franka_moveit.launch"

# TODO: A exit sequence for identifying failed initialization or broken Gazebo
exec "$@"
