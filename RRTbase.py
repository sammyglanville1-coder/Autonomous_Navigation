# RRTbase.py - Fixed version with pixel ↔ world conversion and obstacle inflation
import math
import random
import cv2
import numpy as np
import os
import yaml


def load_map_metadata(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    resolution = data['resolution']  # meters per pixel
    origin = data['origin']  # [x, y, theta]
    return resolution, origin


class RRTGraph:
    def __init__(self, start, goal, map_path, yaml_path):
        if not os.path.isfile(map_path):
            raise FileNotFoundError(f"Map not found: {map_path}")
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"YAML not found: {yaml_path}")

        self.map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        self.map_img = cv2.flip(self.map_img, 0)  # Match RViz orientation
        self.map_height, self.map_width = self.map_img.shape

        self.resolution, self.origin = load_map_metadata(yaml_path)

        # Convert start/goal from meters to pixels
        self.start = self.world_to_pixel(start)
        self.goal = self.world_to_pixel(goal)

        self.x = [self.start[0]]
        self.y = [self.start[1]]
        self.parent = [0]

        self.goal_flag = False
        self.goal_state = None
        self.path = []

        # Inflate obstacles for robot size
        robot_radius_px = int(0.2 / self.resolution)  # Example: 20 cm radius
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (robot_radius_px * 2, robot_radius_px * 2))
        self.map_img = cv2.erode(self.map_img, kernel)  # Shrink free space

    def world_to_pixel(self, pos):
        x_m, y_m = pos
        px = int((x_m - self.origin[0]) / self.resolution)
        py = int((y_m - self.origin[1]) / self.resolution)
        return (px, py)

    def pixel_to_world(self, pos):
        px, py = pos
        x_m = px * self.resolution + self.origin[0]
        y_m = py * self.resolution + self.origin[1]
        return (x_m, y_m)

    def is_free(self, x, y):
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            return self.map_img[int(y), int(x)] > 250
        return False

    def collision_free(self, x1, y1, x2, y2):
        steps = int(max(abs(x2 - x1), abs(y2 - y1)))
        if steps == 0:
            return self.is_free(x1, y1)
        for i in range(steps + 1):
            u = i / steps
            x = int(x1 * (1 - u) + x2 * u)
            y = int(y1 * (1 - u) + y2 * u)
            if not self.is_free(x, y):
                return False
        return True

    def add_node(self, n, x, y):
        self.x.insert(n, x)
        self.y.insert(n, y)

    def remove_node(self, n):
        self.x.pop(n)
        self.y.pop(n)

    def add_edge(self, parent, child):
        self.parent.insert(child, parent)

    def number_of_nodes(self):
        return len(self.x)

    def distance(self, n1, n2):
        x1, y1 = self.x[n1], self.y[n1]
        x2, y2 = self.x[n2], self.y[n2]
        return math.hypot(x2 - x1, y2 - y1)

    def nearest(self, n):
        distances = [self.distance(i, n) for i in range(n)]
        return distances.index(min(distances))

    def sample_environment(self):
        while True:
            x = int(random.uniform(0, self.map_width))
            y = int(random.uniform(0, self.map_height))
            if self.is_free(x, y):
                return x, y

    def step(self, n_near, n_rand, d_max=35):
        d = self.distance(n_near, n_rand)
        if d > d_max:
            x_near, y_near = self.x[n_near], self.y[n_near]
            x_rand, y_rand = self.x[n_rand], self.y[n_rand]
            theta = math.atan2(y_rand - y_near, x_rand - x_near)
            x = int(x_near + d_max * math.cos(theta))
            y = int(y_near + d_max * math.sin(theta))
            self.remove_node(n_rand)
            if self.is_free(x, y):
                if math.hypot(x - self.goal[0], y - self.goal[1]) < d_max:
                    self.add_node(n_rand, self.goal[0], self.goal[1])
                    self.goal_state = n_rand
                    self.goal_flag = True
                else:
                    self.add_node(n_rand, x, y)

    def connect(self, n1, n2):
        x1, y1 = self.x[n1], self.y[n1]
        x2, y2 = self.x[n2], self.y[n2]
        if self.collision_free(x1, y1, x2, y2):
            self.add_edge(n1, n2)
            return True
        else:
            self.remove_node(n2)
            return False

    def expand(self):
        n = self.number_of_nodes()
        x, y = self.sample_environment()
        self.add_node(n, x, y)
        if not self.is_free(x, y):
            return self.x, self.y, self.parent  # Node was removed, skip

        n_near = self.nearest(n)
        self.step(n_near, n)
        if n < self.number_of_nodes():  # Ensure node still exists after step
            self.connect(n_near, n)
        return self.x, self.y, self.parent

    def path_to_goal(self):
        if self.goal_flag and self.goal_state is not None and self.goal_state < len(self.parent):
            self.path = [self.goal_state]
            new_pos = self.parent[self.goal_state]
            while new_pos != 0:
                self.path.append(new_pos)
                new_pos = self.parent[new_pos]
            self.path.append(0)
            return True
        return False

    def getPathCoords(self):
        return [(self.x[i], self.y[i]) for i in self.path]
