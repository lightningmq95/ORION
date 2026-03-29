#!/usr/bin/env python3

import math
import cv2
import numpy as np
import heapq
from std_msgs.msg import Header
from nav_msgs.msg import GridCells, OccupancyGrid, Path
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped

DIRECTIONS_OF_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS_OF_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

class PriorityQueue:
    def __init__(self):
        self.elements = []
    def empty(self) -> bool:
        return not self.elements
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    def get(self):
        return heapq.heappop(self.elements)[1]

def quaternion_from_euler(ai, aj, ak):
    q = [0.0] * 4
    cy = math.cos(ak * 0.5)
    sy = math.sin(ak * 0.5)
    cp = math.cos(aj * 0.5)
    sp = math.sin(aj * 0.5)
    cr = math.cos(ai * 0.5)
    sr = math.sin(ai * 0.5)
    q[0] = sr * cp * cy - cr * sp * sy
    q[1] = cr * sp * cy + sr * cp * sy
    q[2] = cr * cp * sy - sr * sp * cy
    q[3] = cr * cp * cy + sr * sp * sy
    return q

class PathPlanner:
    @staticmethod
    def grid_to_index(mapdata: OccupancyGrid, p: "tuple[int, int]") -> int:
        return p[1] * mapdata.info.width + p[0]

    @staticmethod
    def get_cell_value(mapdata: OccupancyGrid, p: "tuple[int, int]") -> int:
        return mapdata.data[PathPlanner.grid_to_index(mapdata, p)]

    @staticmethod
    def euclidean_distance(p1: "tuple[float, float]", p2: "tuple[float, float]") -> float:
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def grid_to_world(mapdata: OccupancyGrid, p: "tuple[int, int]") -> Point:
        x = (p[0] + 0.5) * mapdata.info.resolution + mapdata.info.origin.position.x
        y = (p[1] + 0.5) * mapdata.info.resolution + mapdata.info.origin.position.y
        return Point(x=float(x), y=float(y), z=0.0)

    @staticmethod
    def world_to_grid(mapdata: OccupancyGrid, wp: Point) -> "tuple[int, int]":
        x = int((wp.x - mapdata.info.origin.position.x) / mapdata.info.resolution)
        y = int((wp.y - mapdata.info.origin.position.y) / mapdata.info.resolution)
        return (x, y)

    @staticmethod
    def path_to_poses(mapdata: OccupancyGrid, path: "list[tuple[int, int]]") -> "list[PoseStamped]":
        poses = []
        for i in range(len(path) - 1):
            cell = path[i]
            next_cell = path[i + 1]
            angle_to_next = 0.0
            if i != len(path) - 1:
                angle_to_next = math.atan2(next_cell[1] - cell[1], next_cell[0] - cell[0])
            q = quaternion_from_euler(0, 0, angle_to_next)
            
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "odom"
            pose_stamped.pose.position = PathPlanner.grid_to_world(mapdata, cell)
            pose_stamped.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            poses.append(pose_stamped)
        return poses

    @staticmethod
    def is_cell_in_bounds(mapdata: OccupancyGrid, p: "tuple[int, int]") -> bool:
        return 0 <= p[0] < mapdata.info.width and 0 <= p[1] < mapdata.info.height

    @staticmethod
    def is_cell_walkable(mapdata: OccupancyGrid, p: "tuple[int, int]") -> bool:
        if not PathPlanner.is_cell_in_bounds(mapdata, p): return False
        val = PathPlanner.get_cell_value(mapdata, p)
        return 0 <= val < 50

    @staticmethod
    def neighbors(mapdata, p, directions, must_be_walkable=True):
        neighbors = []
        for direction in directions:
            candidate = (p[0] + direction[0], p[1] + direction[1])
            if (must_be_walkable and PathPlanner.is_cell_walkable(mapdata, candidate)) or \
               (not must_be_walkable and PathPlanner.is_cell_in_bounds(mapdata, candidate)):
                neighbors.append(candidate)
        return neighbors

    @staticmethod
    def neighbors_of_4(mapdata, p, must_be_walkable=True):
        return PathPlanner.neighbors(mapdata, p, DIRECTIONS_OF_4, must_be_walkable)

    @staticmethod
    def neighbors_of_8(mapdata, p, must_be_walkable=True):
        return PathPlanner.neighbors(mapdata, p, DIRECTIONS_OF_8, must_be_walkable)

    @staticmethod
    def neighbors_and_distances_of_8(mapdata, p, must_be_walkable=True):
        neighbors = []
        for direction in DIRECTIONS_OF_8:
            candidate = (p[0] + direction[0], p[1] + direction[1])
            if not must_be_walkable or PathPlanner.is_cell_walkable(mapdata, candidate):
                distance = PathPlanner.euclidean_distance(direction, (0, 0))
                neighbors.append((candidate, distance))
        return neighbors

    @staticmethod
    def calc_cspace(mapdata: OccupancyGrid):
        KERNEL_SIZE = 15
        
        width = mapdata.info.width
        height = mapdata.info.height
        
        map_arr = np.array(mapdata.data).reshape(height, width).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
        
        unknown_area_mask = cv2.inRange(map_arr, 255, 255)
        unknown_area_mask = cv2.erode(unknown_area_mask, kernel, iterations=1)
        
        map_arr[map_arr == 255] = 0
        obstacle_mask = cv2.dilate(map_arr, kernel, iterations=1)
        
        cspace_data = cv2.bitwise_or(obstacle_mask, unknown_area_mask)
        
        final_cspace_array = np.where(cspace_data > 0, 100, 0).astype(np.int8)
        
        cspace = OccupancyGrid()
        cspace.header = mapdata.header
        cspace.info = mapdata.info
        cspace.data = final_cspace_array.flatten().tolist()
        return cspace

    @staticmethod
    def calc_cost_map(mapdata: OccupancyGrid) -> np.ndarray:
        width = mapdata.info.width
        height = mapdata.info.height
        map_arr = np.array(mapdata.data).reshape(height, width).astype(np.uint8)
        map_arr[map_arr == 255] = 100

        cost_map = np.zeros_like(map_arr)
        dilated_map = map_arr.copy()
        iterations = 0
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        
        while np.any(dilated_map == 0) and iterations < 100:
            iterations += 1
            next_dilated_map = cv2.dilate(dilated_map, kernel, iterations=1)
            difference = next_dilated_map - dilated_map
            difference[difference > 0] = iterations
            cost_map = cv2.bitwise_or(cost_map, difference)
            dilated_map = next_dilated_map

        cost_map[cost_map > 0] -= 1
        return cost_map

    @staticmethod
    def a_star(mapdata, cost_map, start, goal):
        COST_MAP_WEIGHT = 100 # additional penalty to stay away from walls
        if not PathPlanner.is_cell_walkable(mapdata, start): return (None, None, start, goal)
        if not PathPlanner.is_cell_walkable(mapdata, goal): return (None, None, start, goal)

        pq = PriorityQueue()
        pq.put(start, 0)

        cost_so_far = {start: 0}
        distance_cost_so_far = {start: 0}
        came_from = {start: None}

        while not pq.empty():
            current = pq.get()
            if current == goal: break

            for neighbor, distance in PathPlanner.neighbors_and_distances_of_8(mapdata, current):
                added_cost = distance + COST_MAP_WEIGHT * cost_map[neighbor[1]][neighbor[0]]
                new_cost = cost_so_far[current] + added_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    distance_cost_so_far[neighbor] = distance_cost_so_far[current] + distance
                    priority = new_cost + PathPlanner.euclidean_distance(neighbor, goal)
                    pq.put(neighbor, priority)
                    came_from[neighbor] = current

        path = []
        cell = goal
        while cell:
            path.insert(0, cell)
            if cell in came_from:
                cell = came_from[cell]
            else:
                return (None, None, start, goal)

        if len(path) < 5: 
            return (None, None, start, goal)
            
        chop_length = min(4, len(path) - 1)
        if chop_length > 0:
            path = path[:-chop_length]
            
        return (path, distance_cost_so_far[goal], start, goal)

    @staticmethod
    def path_to_message(mapdata: OccupancyGrid, path: "list[tuple[int, int]]", timestamp) -> Path:
        poses = PathPlanner.path_to_poses(mapdata, path)
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = timestamp
        path_msg.poses = poses
        return path_msg