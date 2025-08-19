# Autonomous Navigation with RRT + NMPC (Ackermann, ROS)
This project implements an **autonomous navigation system** for an Ackermann-steered JetAcker robot.  
It combines **RRT (Rapidly-exploring Random Tree)** for **global path planning** with  
**NMPC (Nonlinear Model Predictive Control)** for **local trajectory optimization**.  

The **intermediate nodes from the RRT path** are passed **iteratively into the NMPC solver**,  
so the robot follows the global path while **respecting kinematic constraints** of Ackermann steering.  



##  Features
- Global path planning with **RRT / RRT***  
- Local control with **NMPC (OpEn + CasADi)**  
- **ROS Noetic integration** with RViz visualization  
- Supports both **forward and reverse motion**  
- Works with **LiDAR-based maps (map.pgm/map.yaml)**  
- Modular nodes: RRT planner, NMPC controller, optimizer builder
- 
## Tested On
- ROS Noetic (Ubuntu 20.04, WSL)
- Python 3.8
- OpEn (Optimization Engine) + CasADi

## Rviz setup
/initialpose → geometry_msgs/PoseWithCovarianceStamped

/move_base_simple/goal → geometry_msgs/PoseStamped

/rrt_path → nav_msgs/Path

/nmpc_path → nav_msgs/Path



/robot_pose → geometry_msgs/PoseStamped

#to laucnh everything use;
roslaunch rrt_mapping system.launch



## Quick Start

## Install
```bash
## Python deps
pip install -r requirements.txt

# ROS deps (Noetic)
sudo apt install ros-noetic-rospy ros-noetic-geometry-msgs ros-noetic-nav-msgs \
                 ros-noetic-std-msgs ros-noetic-tf ros-noetic-rviz \
                 ros-noetic-map-server ros-noetic-robot-state-publisher

