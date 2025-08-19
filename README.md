# Autonomous_Navigation
Autonomous navigation system combining RRT for global path planning with NMPC for local trajectory optimization. Intermediate RRT nodes are passed iteratively into the NMPC solver so the robot follows the path while respecting Ackermann vehicle kinematic constraints. Built for ROS with LiDAR-based mapping and RViz visualization.
