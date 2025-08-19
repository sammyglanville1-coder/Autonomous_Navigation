#!/usr/bin/env python3
import sys
import math
import csv
import time
import rospy
import tf
import matplotlib.pyplot as plt

from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler

# Optimizer
sys.path.insert(1, "/root/Project3/src/rrt_mapping/scripts/my_optimizers/navigation_hybrid/navigation_hybrid")
import navigation_hybrid


class RRTtoNMPC:
    def __init__(self):
        rospy.init_node("rrt_to_nmpc", anonymous=True)
        rospy.loginfo("RRT → NMPC Hybrid Node Started")

        # Parameters
        self.time_step = 0.1
        self.wheelbase = 0.214
        self.segment_tolerance = 0.8
        self.goal_tolerance = 0.3
        self.yaw_tolerance = math.radians(20)

        # State
        self.robot_state = None  # [x, y, yaw]
        self.initial_yaw = None
        self.rrt_waypoints = []
        self.start_pose_received = False

        # Logging
        self.log_data = []
        self.segment_start_time = time.time()
        self.total_nmpc_time = 0
        self.reverse_count = 0

        # ROS Interface
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.solver = navigation_hybrid.solver()

        self.path_pub = rospy.Publisher("/nmpc_path", Path, queue_size=10)
        self.pose_pub = rospy.Publisher("/robot_pose", PoseStamped, queue_size=10)

        rospy.Subscriber("/rrt_path", Path, self.on_rrt_path)
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.on_initial_pose)

        rospy.spin()

    # Callbacks
    def on_initial_pose(self, msg):
        pos = msg.pose.pose.position
        yaw = self.yaw_from_quaternion(msg.pose.pose.orientation)

        self.robot_state = [pos.x, pos.y, yaw]
        self.initial_yaw = yaw
        self.start_pose_received = True

        rospy.loginfo(f"Initial pose: x={pos.x:.2f}, y={pos.y:.2f}, yaw={math.degrees(yaw):.2f}")

    def on_rrt_path(self, msg):
        if not self.start_pose_received:
            rospy.logwarn("Start pose not set. Use '2D Pose Estimate' in RViz first.")
            return

        self.rrt_waypoints = self.extract_waypoints(msg)

        if len(self.rrt_waypoints) < 2:
            rospy.logwarn("RRT path has too few waypoints.")
            return

        rospy.loginfo(f"RRT path received with {len(self.rrt_waypoints)} waypoints.")
        self.follow_full_path()

    # Core Path Following
    def follow_full_path(self):
        combined_path = Path()
        combined_path.header.frame_id = "map"

        for i, waypoint in enumerate(self.rrt_waypoints[1:], start=1):
            rospy.loginfo(f"Waypoint {i}/{len(self.rrt_waypoints) - 1}: {waypoint}")
            self.follow_segment(i, waypoint, combined_path, is_final=(i == len(self.rrt_waypoints) - 1))

        rospy.loginfo("NMPC completed the RRT path.")
        self.path_pub.publish(combined_path)

        # Summary metrics
        steps = len(self.log_data)
        segments = max(len(self.rrt_waypoints) - 1, 0)
        avg_step_time = (self.total_nmpc_time / steps) if steps > 0 else 0.0
        avg_segment_time = (self.total_nmpc_time / segments) if segments > 0 else 0.0
        final_yaw_error_deg = math.degrees(self.log_data[-1]["yaw_error"]) if self.log_data else 0.0

        path_length_m = 0.0
        if steps > 1:
            for i in range(steps - 1):
                dx = self.log_data[i+1]["x"] - self.log_data[i]["x"]
                dy = self.log_data[i+1]["y"] - self.log_data[i]["y"]
                path_length_m += math.hypot(dx, dy)

        summary = {
            "segments": segments,
            "steps": steps,
            "total_time_s": round(self.total_nmpc_time, 6),
            "avg_step_time_s": round(avg_step_time, 6),
            "avg_segment_time_s": round(avg_segment_time, 6),
            "reverse_count": self.reverse_count,
            "final_yaw_error_deg": round(final_yaw_error_deg, 6),
            "path_length_m": round(path_length_m, 6),
        }

        rospy.loginfo(f"Average NMPC segment time: {avg_segment_time:.4f}s")
        rospy.loginfo(f"Final yaw error: {final_yaw_error_deg:.2f}°")

        # Save logs and plots
        self.save_csv_log("/tmp/nmpc_log.csv", summary=summary)
        self.plot_path("/tmp/nmpc_path.png")

    def follow_segment(self, index, target, path_msg, is_final=False):
        max_steps = 500
        steps = 0

        while not rospy.is_shutdown() and steps < max_steps:
            if self.target_reached(target, is_final):
                break

            if is_final:
                target_yaw = target[2]  # exact final yaw from RViz
            else:
                target_yaw = self.compute_lookahead_yaw(index)

            # Only use initial yaw for the very first step
            if index == 1 and steps == 0:
                solver_input = [self.robot_state[0], self.robot_state[1], self.initial_yaw,
                                target[0], target[1], target_yaw]
            else:
                solver_input = self.robot_state + [target[0], target[1], target_yaw]

            control_cmd = self.solve_nmpc_with_input(solver_input)

            if not control_cmd:
                break

            self.apply_motion(control_cmd)
            self.publish_pose(path_msg)
            self.log_step(control_cmd, target)

            steps += 1
            rospy.sleep(self.time_step)

    def extract_waypoints(self, path_msg):
        poses_reversed = list(reversed(path_msg.poses))
        waypoints = []

        for i, pose_stamped in enumerate(poses_reversed):
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y

            if i < len(poses_reversed) - 1:
                next_pose = poses_reversed[i + 1].pose.position
                dx = next_pose.x - x
                dy = next_pose.y - y
                yaw = math.atan2(dy, dx)
            else:
                yaw = self.yaw_from_quaternion(pose_stamped.pose.orientation)

            waypoints.append((x, y, yaw))

        return waypoints

    def target_reached(self, target, is_final):
        dist = math.hypot(self.robot_state[0] - target[0], self.robot_state[1] - target[1])

        if is_final:
            yaw_error = self.normalize_angle(target[2] - self.robot_state[2])
            if dist < self.goal_tolerance and abs(yaw_error) < self.yaw_tolerance:
                rospy.loginfo("Final goal reached with yaw alignment.")
                return True
        else:
            if dist < self.segment_tolerance:
                rospy.loginfo("Intermediate waypoint reached.")
                return True
        return False

    def compute_lookahead_yaw(self, index):
        lookahead_idx = min(index + 3, len(self.rrt_waypoints) - 1)
        return math.atan2(
            self.rrt_waypoints[lookahead_idx][1] - self.robot_state[1],
            self.rrt_waypoints[lookahead_idx][0] - self.robot_state[0]
        )

    def solve_nmpc_with_input(self, solver_input):
        result = self.solver.run(solver_input).solution
        if result is None or len(result) < 2:
            rospy.logwarn("NMPC returned no valid solution.")
            return None
        return result

    def apply_motion(self, control_cmd):
        v, delta = control_cmd[0], control_cmd[1]

        x, y, theta = self.robot_state
        theta_dot = v * math.tan(delta) / self.wheelbase

        x += self.time_step * v * math.cos(theta)
        y += self.time_step * v * math.sin(theta)
        theta += self.time_step * theta_dot

        self.robot_state = [x, y, theta]
        self.tf_broadcaster.sendTransform(
            (x, y, 0.0),
            quaternion_from_euler(0, 0, theta),
            rospy.Time.now(),
            "base_link",
            "map"
        )

    def publish_pose(self, path_msg):
        x, y, theta = self.robot_state
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, theta))

        self.pose_pub.publish(pose)
        path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def log_step(self, control_cmd, target):
        v, delta = control_cmd[0], control_cmd[1]
        dist_to_target = math.hypot(self.robot_state[0] - target[0], self.robot_state[1] - target[1])

        segment_time = time.time() - self.segment_start_time
        self.total_nmpc_time += segment_time
        self.segment_start_time = time.time()

        self.log_data.append({
            "x": self.robot_state[0],
            "y": self.robot_state[1],
            "theta": self.robot_state[2],
            "v": v,
            "delta": delta,
            "is_reverse": int(v < 0),
            "yaw_error": self.normalize_angle(target[2] - self.robot_state[2]),
            "dist_to_target": dist_to_target,
            "segment_time": segment_time
        })

        if v < 0:
            self.reverse_count += 1

    def save_csv_log(self, filepath, summary=None):
        if not self.log_data:
            return

        with open(filepath, "w", newline='') as f:
            fieldnames = list(self.log_data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.log_data)

            if summary:
                f.write("\nSUMMARY,VALUE\n")
                for k, v in summary.items():
                    f.write(f"{k},{v}\n")

        rospy.loginfo(f"NMPC log saved to {filepath}")
        rospy.loginfo(f"Total time: {self.total_nmpc_time:.2f}s, Reverse count: {self.reverse_count}")

    def plot_path(self, filepath):
        xs = [row["x"] for row in self.log_data]
        ys = [row["y"] for row in self.log_data]
        plt.figure()
        plt.plot(xs, ys, marker='o')
        plt.title("NMPC Trajectory")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.grid(True)
        plt.savefig(filepath)
        rospy.loginfo(f"Path plot saved to {filepath}")

    def yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))


if __name__ == "__main__":
    try:
        RRTtoNMPC()
    except rospy.ROSInterruptException:
        pass
