#!/usr/bin/env python3
import sys
import os
import math
import csv
import time
import rospy
import matplotlib.pyplot as plt

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_euler, euler_from_quaternion

# Import RRTGraph from same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RRTbase import RRTGraph


class RRTGoalPlanner:
    def __init__(self):
        rospy.init_node("rrt_goal_planner")

        # Subscribers
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.on_start_pose)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.on_goal_pose)

        # Publishers
        self.tree_pub = rospy.Publisher("/rrt_tree", Marker, queue_size=1)
        self.path_pub = rospy.Publisher("/rrt_path", Path, queue_size=1)

        # NEW: thick path marker publisher
        self.path_line_pub = rospy.Publisher("/rrt_path_marker", Marker, queue_size=1)

        # Planner settings
        self.start_pose = None
        self.goal_pose = None
        self.map_path = os.path.join(os.path.dirname(__file__), "../maps/map.pgm")
        self.map_yaml = os.path.join(os.path.dirname(__file__), "../maps/map.yaml")
        self.max_iterations = 5000

        # NEW: configurable path line width (meters) and color
        self.path_line_width = rospy.get_param("~path_line_width", 20)  # thickness in meters
        self.path_color_rgba = rospy.get_param("~path_color_rgba", [1.0, 0.2, 0.2, 1.0])  # RGBA

        rospy.loginfo("Waiting for start and goal to be set in RViz...")
        rospy.spin()

    def on_start_pose(self, msg):
        self.start_pose = self._pose_to_tuple(msg.pose.pose)
        rospy.loginfo(f"Start pose set: {self.start_pose}")
        self.try_plan_path()

    def on_goal_pose(self, msg):
        self.goal_pose = self._pose_to_tuple(msg.pose)
        rospy.loginfo(f"Goal pose set: {self.goal_pose}")
        self.try_plan_path()

    def try_plan_path(self):
        if not self.start_pose or not self.goal_pose:
            return

        rospy.loginfo("Building RRT tree...")
        rrt = RRTGraph(self.start_pose[:2], self.goal_pose[:2], self.map_path, self.map_yaml)

        start_time = time.time()
        found = False
        iterations_used = 0

        for i in range(self.max_iterations):
            rrt.expand()
            if rrt.path_to_goal():
                rospy.loginfo("Path found!")
                found = True
                iterations_used = i + 1
                break

        end_time = time.time()
        planning_time = end_time - start_time

        if not found:
            rospy.logwarn("No path found after max iterations.")
            return

        # Convert path to world coords
        path_coords = [rrt.pixel_to_world((px, py)) for px, py in rrt.getPathCoords()]
        path_length = sum(
            math.hypot(path_coords[i + 1][0] - path_coords[i][0],
                       path_coords[i + 1][1] - path_coords[i][1])
            for i in range(len(path_coords) - 1)
        )
        avg_spacing = path_length / (len(path_coords) - 1) if len(path_coords) > 1 else 0.0

        num_nodes = rrt.number_of_nodes()

        # Publish tree + path
        self.publish_tree(rrt)
        self.publish_path(rrt)  # publishes both nav_msgs/Path and the thick Marker

        # Save CSV log (overwriting each run)
        try:
            with open("/tmp/rrt_log.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                for x, y in path_coords:
                    writer.writerow([x, y])
                f.write("\nSUMMARY,VALUE\n")
                f.write(f"planning_time_s,{planning_time}\n")
                f.write(f"iterations,{iterations_used}\n")
                f.write(f"nodes,{num_nodes}\n")
                f.write(f"path_length_m,{path_length}\n")
                f.write(f"avg_spacing_m,{avg_spacing}\n")
            rospy.loginfo("RRT path log saved to /tmp/rrt_log.csv")
        except Exception as e:
            rospy.logwarn(f"Failed to save RRT log: {e}")

        # Save path plot (overwrite)
        try:
            xs, ys = zip(*path_coords)
            plt.figure()
            plt.plot(xs, ys, marker='o', linewidth=3)  # CHANGED: thicker line in PNG
            plt.title("RRT Path")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.legend(["RRT Path"])
            plt.grid(True)
            plt.savefig("/tmp/rrt_path.png")
            plt.close()
            rospy.loginfo("RRT path plot saved to /tmp/rrt_path.png")
        except Exception as e:
            rospy.logwarn(f"Failed to save RRT plot: {e}")

    def publish_tree(self, rrt):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "rrt_tree"
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # thickness of tree edges; increase if you want thicker tree
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
        marker.points = []

        for i in range(1, rrt.number_of_nodes()):
            parent_idx = rrt.parent[i]
            p1 = Point(*rrt.pixel_to_world((rrt.x[i], rrt.y[i])), 0.0)
            p2 = Point(*rrt.pixel_to_world((rrt.x[parent_idx], rrt.y[parent_idx])), 0.0)
            marker.points.extend([p1, p2])

        self.tree_pub.publish(marker)
        rospy.loginfo("RRT tree published.")

    def publish_path(self, rrt):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        coords = [rrt.pixel_to_world((px, py)) for px, py in rrt.getPathCoords()]

        for i, (x, y) in enumerate(coords):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y

            if i == 0:
                yaw = self.start_pose[2]
            elif i < len(coords) - 1:
                dx = coords[i + 1][0] - x
                dy = coords[i + 1][1] - y
                yaw = math.atan2(dy, dx)
            else:
                yaw = self.goal_pose[2]

            q = quaternion_from_euler(0, 0, yaw)
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
            path_msg.poses.append(pose)

        # Publish the standard Path
        self.path_pub.publish(path_msg)
        rospy.loginfo("Final path published to /rrt_path")

        # NEW: Publish a thick Marker LINE_STRIP for the same path
        if coords:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "rrt_path"
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # Thickness in meters
            marker.scale.x = float(self.path_line_width)
            marker.scale.x = 2
            # Color RGBA
            r, g, b, a = self.path_color_rgba if len(self.path_color_rgba) == 4 else [1.0, 0.2, 0.2, 1.0]
            marker.color = ColorRGBA(float(r), float(g), float(b), float(a))

            # Points along the path
            marker.points = [Point(x, y, 0.0) for (x, y) in coords]

            self.path_line_pub.publish(marker)
            rospy.loginfo("Thick path marker published to /rrt_path_marker")

    def _pose_to_tuple(self, pose):
        x, y = pose.position.x, pose.position.y
        yaw = euler_from_quaternion([pose.orientation.x, pose.orientation.y,
                                     pose.orientation.z, pose.orientation.w])[2]
        return (x, y, yaw)


if __name__ == '__main__':
    RRTGoalPlanner()
