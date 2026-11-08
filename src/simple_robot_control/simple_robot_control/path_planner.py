# Authors: Akash Mohapatra, Manas Chintawar

import functools
import heapq
import math

import cv2

from geometry_msgs.msg import Point, PoseStamped, Quaternion

from nav_msgs.msg import OccupancyGrid, Path

import numpy as np

DIRECTIONS_OF_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS_OF_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                   (1, -1), (1, 0), (1, 1)]
_SQRT2 = math.sqrt(2)
DIRECTIONS_AND_DISTANCES_OF_8 = [
    (direction, _SQRT2 if direction[0] and direction[1] else 1.0)
    for direction in DIRECTIONS_OF_8
]


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
    cy, sy = math.cos(ak * 0.5), math.sin(ak * 0.5)
    cp, sp = math.cos(aj * 0.5), math.sin(aj * 0.5)
    cr, sr = math.cos(ai * 0.5), math.sin(ai * 0.5)
    return [sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy]


def _map_array(mapdata, grid=None):
    width, height = int(mapdata.info.width), int(mapdata.info.height)
    if width <= 0 or height <= 0:
        raise ValueError('Occupancy grid dimensions must be positive')
    data = mapdata.data if grid is None else grid
    return np.asarray(data, dtype=np.int8).reshape(height, width)


@functools.lru_cache(maxsize=16)
def _inflation_kernel(kernel_size):
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


class PlanningGrid:
    """
    Prepared graph shared by all searches on one C-space/cost-map pair.

    A blocked border eliminates bounds checks in the expansion loop. Python
    scalars avoid NumPy scalar conversion on every relaxation. Search storage
    is reused, with generation marks instead of clearing full maps per goal.
    """

    def __init__(self, mapdata, cost_map, grid=None):
        self.width = int(mapdata.info.width)
        self.height = int(mapdata.info.height)
        self.stride = self.width + 2
        grid = _map_array(mapdata, grid)
        shape = grid.shape
        cost_map = np.asarray(cost_map)
        if cost_map.shape != shape:
            raise ValueError('Cost map shape must match the occupancy grid')
        if not np.isfinite(cost_map).all() or np.any(cost_map < 0):
            raise ValueError('Cost map values must be finite and nonnegative')

        free = (grid >= 0) & (grid < 50)
        self._components = cv2.connectedComponents(
            free.astype(np.uint8), connectivity=4)[1]
        walkable = np.pad(free, 1)
        self._walkable = walkable.tobytes()
        self._has_penalties = bool(np.any(cost_map[free]))
        penalties = np.pad(cost_map.astype(np.float64) * 0.5, 1)
        self._penalties = penalties.ravel().tolist()
        size = len(self._walkable)
        self._costs = [math.inf] * size
        self._parents = [-1] * size
        self._generations = [0] * size
        self._generation = 0
        stride = self.stride
        # offset, movement cost, orthogonal checks for diagonal moves
        self._steps = (
            (-stride - 1, _SQRT2, -stride, -1),
            (-stride, 1.0, 0, 0),
            (-stride + 1, _SQRT2, -stride, 1),
            (-1, 1.0, 0, 0), (1, 1.0, 0, 0),
            (stride - 1, _SQRT2, stride, -1),
            (stride, 1.0, 0, 0),
            (stride + 1, _SQRT2, stride, 1),
        )

    def is_walkable(self, cell):
        """Check whether a cell is in bounds and traversable."""
        x, y = cell
        return (0 <= x < self.width and 0 <= y < self.height
                and bool(self._walkable[(y + 1) * self.stride + x + 1]))

    def reachable_mask(self, start):
        """Return cells connected to start without cutting obstacle corners."""
        if not self.is_walkable(start):
            return np.zeros(self._components.shape, dtype=bool)
        return self._components == self._components[start[1], start[0]]

    def _index(self, cell):
        return (cell[1] + 1) * self.stride + cell[0] + 1

    def a_star(self, start, goal, *, use_cost_map=True):
        """Find a route, optionally minimizing distance without soft costs."""
        return self._search(start, [goal], use_cost_map=use_cost_map)[0]

    def a_star_many(self, start, goals):
        """
        Plan targets together, preserving order and geometric length costs.

        Weighted maps share graph expansions. Zero-penalty maps use separate
        goal-directed searches, which prune more aggressively in open space.
        Each search uses an admissible, consistent octile heuristic and
        returns the minimum weighted-cost route to each reachable goal.
        """
        goals = list(goals)
        # In open, zero-penalty maps, goal-directed searches avoid sweeping
        # the entire rectangle between widely separated frontiers.
        if not self._has_penalties and len(goals) > 1:
            return [self.a_star(start, goal) for goal in goals]
        return self._search(start, goals)

    def _search(self, start, goals, *, use_cost_map=True):
        """
        Find minimum weighted-cost paths to all goals in one search.

        The octile distance to the goals' bounding box is an admissible,
        consistent heuristic for every target. Unlike repeated A*, shared
        prefixes are expanded once. Results retain input order and report
        geometric path lengths, as the original a_star API does.
        """
        goals = list(goals)
        results = [(None, None, start, goal) for goal in goals]
        if not goals or not self.is_walkable(start):
            return results
        targets = {}
        component = self._components[start[1], start[0]]
        for i, goal in enumerate(goals):
            if (self.is_walkable(goal)
                    and self._components[goal[1], goal[0]] == component):
                targets.setdefault(self._index(goal), []).append(i)
        has_penalties = use_cost_map and self._has_penalties
        if not has_penalties:
            for target, indices in list(targets.items()):
                direct = self._direct_path(start, goals[indices[0]])
                if direct is not None:
                    path, distance = direct
                    for i in indices:
                        results[i] = (path, distance, start, goals[i])
                    del targets[target]
        if not targets:
            return results

        # A rectangle gives a constant-time heuristic even with many targets.
        valid_goals = [goals[indices[0]] for indices in targets.values()]
        min_x = min(p[0] for p in valid_goals) + 1
        max_x = max(p[0] for p in valid_goals) + 1
        min_y = min(p[1] for p in valid_goals) + 1
        max_y = max(p[1] for p in valid_goals) + 1
        stride = self.stride
        walkable = self._walkable
        penalties = self._penalties
        costs = self._costs
        parents = self._parents
        generations = self._generations
        self._generation += 1
        generation = self._generation
        start_index = self._index(start)
        costs[start_index] = 0.0
        generations[start_index] = generation
        parents[start_index] = -1
        queue = [(0.0, 0.0, start_index)]
        push, pop = heapq.heappush, heapq.heappop
        steps = self._steps
        diagonal_extra = _SQRT2 - 1.0

        while queue and targets:
            _, negative_cost, current = pop(queue)
            current_cost = -negative_cost
            if current_cost != costs[current]:
                continue  # discard stale heap entries, never re-expand them
            if current in targets:
                path, distance = self._reconstruct(current)
                for i in targets.pop(current):
                    results[i] = (path, distance, start, goals[i])
                if not targets:
                    break

            for offset, distance, side1, side2 in steps:
                neighbor = current + offset
                if not walkable[neighbor]:
                    continue
                if side1 and not (walkable[current + side1]
                                  and walkable[current + side2]):
                    continue  # a robot cannot squeeze diagonally through walls
                new_cost = current_cost + distance
                if has_penalties:
                    new_cost += penalties[neighbor]
                if (generations[neighbor] == generation
                        and new_cost >= costs[neighbor]):
                    continue
                generations[neighbor] = generation
                costs[neighbor] = new_cost
                parents[neighbor] = current
                y, x = divmod(neighbor, stride)
                dx = min_x - x if x < min_x else x - max_x if x > max_x else 0
                dy = min_y - y if y < min_y else y - max_y if y > max_y else 0
                heuristic = (dx + diagonal_extra * dy if dx > dy
                             else dy + diagonal_extra * dx)
                # Prefer greater progress on equal f-cost in open space.
                push(queue, (new_cost + heuristic, -new_cost, neighbor))
        return results

    def _direct_path(self, start, goal):
        """Return an octile-optimal clear route, or defer to graph search."""
        x, y = start
        gx, gy = goal
        dx, dy = abs(gx - x), abs(gy - y)
        sx, sy = (1 if x < gx else -1), (1 if y < gy else -1)
        error = dx - dy
        stride, walkable = self.stride, self._walkable
        current = self._index(start)
        path = [start]
        while (x, y) != (gx, gy):
            twice_error = 2 * error
            nx, ny = x, y
            if twice_error >= -dy:
                error -= dy
                nx += sx
            if twice_error <= dx:
                error += dx
                ny += sy
            neighbor = current + (ny - y) * stride + nx - x
            if not walkable[neighbor]:
                return None
            if nx != x and ny != y and not (
                    walkable[current + sx]
                    and walkable[current + sy * stride]):
                return None
            path.append((nx, ny))
            x, y, current = nx, ny, neighbor
        # Monotone Bresenham uses min(dx, dy) diagonals and the minimum number
        # of cardinal moves, so it attains the zero-penalty octile lower bound.
        distance = max(dx, dy) + (_SQRT2 - 1.0) * min(dx, dy)
        return path, distance

    def _reconstruct(self, goal):
        path = []
        distance = 0.0
        current = goal
        while current != -1:
            y, x = divmod(current, self.stride)
            path.append((x - 1, y - 1))
            parent = self._parents[current]
            if parent != -1:
                offset = abs(current - parent)
                distance += 1.0 if offset in (1, self.stride) else _SQRT2
            current = parent
        path.reverse()
        return path, distance


class PathPlanner:

    @staticmethod
    def grid_to_index(mapdata: OccupancyGrid, p: 'tuple[int, int]') -> int:
        return p[1] * mapdata.info.width + p[0]

    @staticmethod
    def get_cell_value(mapdata: OccupancyGrid, p: 'tuple[int, int]') -> int:
        return mapdata.data[PathPlanner.grid_to_index(mapdata, p)]

    @staticmethod
    def euclidean_distance(p1, p2) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def grid_to_world(mapdata: OccupancyGrid, p: 'tuple[int, int]') -> Point:
        resolution = mapdata.info.resolution
        origin = mapdata.info.origin.position
        x = (p[0] + 0.5) * resolution + origin.x
        y = (p[1] + 0.5) * resolution + origin.y
        return Point(x=float(x), y=float(y), z=0.0)

    @staticmethod
    def world_to_grid(mapdata: OccupancyGrid, wp: Point) -> 'tuple[int, int]':
        resolution = mapdata.info.resolution
        origin = mapdata.info.origin.position
        x = math.floor((wp.x - origin.x) / resolution)
        y = math.floor((wp.y - origin.y) / resolution)
        return (x, y)

    @staticmethod
    def path_to_poses(mapdata: OccupancyGrid, path) -> 'list[PoseStamped]':
        poses = []
        frame_id = mapdata.header.frame_id or 'odom'
        angle = 0.0
        for i, cell in enumerate(path):
            if i + 1 < len(path):
                next_cell = path[i + 1]
                angle = math.atan2(
                    next_cell[1] - cell[1], next_cell[0] - cell[0])
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = frame_id
            pose_stamped.pose.position = PathPlanner.grid_to_world(
                mapdata, cell)
            pose_stamped.pose.orientation = Quaternion(
                z=math.sin(angle * 0.5), w=math.cos(angle * 0.5))
            poses.append(pose_stamped)
        return poses

    @staticmethod
    def is_cell_in_bounds(mapdata: OccupancyGrid, p) -> bool:
        return (0 <= p[0] < mapdata.info.width
                and 0 <= p[1] < mapdata.info.height)

    @staticmethod
    def is_cell_walkable(mapdata: OccupancyGrid, p: 'tuple[int, int]') -> bool:
        if not PathPlanner.is_cell_in_bounds(mapdata, p):
            return False
        val = PathPlanner.get_cell_value(mapdata, p)
        return 0 <= val < 50

    @staticmethod
    def check_body_traversability(mapdata, path, body_bounds, margin=0.01):
        """Check continuous forward segments and turns of a rectangular body.

        Paths contain grid cells; bounds are (xmin, xmax, ymin, ymax) in
        metres relative to the robot's reference frame. Occupied cell squares
        must clear the whole body, including its asymmetric front/rear offsets.
        The centreline must remain known free; the body may overhang unknown
        cells at a frontier, consistent with C-space's unknown-space policy.
        A singleton is checked conservatively for every possible heading.
        """
        points = np.asarray(path, dtype=np.float64)
        bounds = np.asarray(body_bounds, dtype=np.float64)
        resolution = float(mapdata.info.resolution)
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
                or bounds.shape != (4,) or not np.isfinite(points).all()
                or not np.isfinite(bounds).all()
                or not math.isfinite(resolution) or resolution <= 0
                or not math.isfinite(margin) or margin < 0):
            return False
        xmin, xmax, ymin, ymax = bounds
        if xmin >= xmax or ymin >= ymax:
            return False
        grid = _map_array(mapdata)
        height, width = grid.shape
        centres = points + 0.5
        if np.any((centres < 0) | (centres >= (width, height))):
            return False
        cells = np.floor(centres).astype(np.intp)
        if np.any((grid[cells[:, 1], cells[:, 0]] < 0)
                  | (grid[cells[:, 1], cells[:, 0]] >= 50)):
            return False
        occupied = grid >= 50
        unknown = grid < 0

        def rectangle_clear(cx, cy, yaw, hx, hy, blocked):
            # Separating-axis intersection of an oriented rectangle with cell
            # squares. Checking just cell centres can miss a wall at an edge.
            c, s = math.cos(yaw), math.sin(yaw)
            abs_c, abs_s = abs(c), abs(s)
            ex, ey = abs_c * hx + abs_s * hy, abs_s * hx + abs_c * hy
            if (cx - ex < 0 or cy - ey < 0
                    or cx + ex > width * resolution
                    or cy + ey > height * resolution):
                return False
            x0 = max(0, math.floor((cx - ex) / resolution - 1e-9))
            y0 = max(0, math.floor((cy - ey) / resolution - 1e-9))
            x1 = min(width, math.floor((cx + ex) / resolution) + 1)
            y1 = min(height, math.floor((cy + ey) / resolution) + 1)
            ys, xs = np.nonzero(blocked[y0:y1, x0:x1])
            if not len(xs):
                return True
            dx = (xs + x0 + 0.5) * resolution - cx
            dy = (ys + y0 + 0.5) * resolution - cy
            half_cell = resolution * 0.5
            projection = half_cell * (abs_c + abs_s)
            overlap = ((np.abs(dx) <= ex + half_cell + 1e-9)
                       & (np.abs(dy) <= ey + half_cell + 1e-9)
                       & (np.abs(c * dx + s * dy) <= hx + projection + 1e-9)
                       & (np.abs(-s * dx + c * dy) <= hy + projection + 1e-9))
            return not np.any(overlap)

        ox, oy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
        hx, hy = (xmax - xmin) * 0.5 + margin, (ymax - ymin) * 0.5 + margin
        radius = math.hypot(abs(ox) + hx, abs(oy) + hy)

        def body_clear(point, yaw, extension=0.0, padding=0.0):
            c, s = math.cos(yaw), math.sin(yaw)
            return rectangle_clear(
                point[0] + c * ox - s * oy,
                point[1] + s * ox + c * oy, yaw,
                hx + extension + padding, hy + padding, occupied)

        # Drop duplicates and merge straight runs so checks scale with bends,
        # rather than with map resolution on long, open corridors.
        points = points[np.r_[True, np.any(np.diff(points, axis=0), axis=1)]]
        if len(points) > 2:
            steps = np.diff(points, axis=0)
            turns = (~np.isclose(
                steps[:-1, 0] * steps[1:, 1],
                steps[:-1, 1] * steps[1:, 0], rtol=1e-10, atol=1e-10)
                     | (np.sum(steps[:-1] * steps[1:], axis=1) <= 0))
            points = points[np.r_[True, turns, True]]
        points = (points + 0.5) * resolution
        if len(points) == 1:
            return rectangle_clear(*points[0], 0.0, radius, radius, occupied)
        steps = np.diff(points, axis=0)
        headings = np.arctan2(steps[:, 1], steps[:, 0])
        for index, (step, yaw) in enumerate(zip(steps, headings)):
            half_length = math.hypot(*step) * 0.5
            midpoint = (points[index] + points[index + 1]) * 0.5
            # Translation along a heading sweeps one elongated rectangle:
            # no gaps between samples, even for sparse input waypoints.
            if (not rectangle_clear(*midpoint, yaw, half_length, 0.0, unknown)
                    or not body_clear(midpoint, yaw, extension=half_length)):
                return False
            if index == 0:
                continue
            turn = math.remainder(yaw - headings[index - 1], 2 * math.pi)
            count = max(1, math.ceil(abs(turn) * radius / (0.5 * resolution)))
            increment = turn / count
            # Each midpoint rectangle is enlarged by the maximum displacement
            # of any body corner during its angular interval, covering the
            # continuous rotation rather than just discrete headings.
            padding = radius * abs(increment) * 0.5
            for sample in range(count):
                angle = headings[index - 1] + (sample + 0.5) * increment
                if not body_clear(points[index], angle, padding=padding):
                    return False
        return True

    @staticmethod
    def resample_path(path):
        """Space continuous grid coordinates at most one cell apart."""
        if not path:
            return []
        result = [tuple(map(float, path[0]))]
        for start, end in zip(path, path[1:]):
            count = max(1, math.ceil(
                PathPlanner.euclidean_distance(start, end)))
            result.extend(map(tuple, np.linspace(start, end, count + 1)[1:]))
        return result

    @staticmethod
    def shortest_path(mapdata, planner, start, goal, body_bounds,
                      margin=0.06):
        """Prefer a straight route, then shorten a distance-based A* route.

        Hard footprint clearance still applies. Soft proximity penalties do
        not make a manual waypoint detour across an otherwise clear room.
        Fractional grid coordinates preserve arbitrary straight-line angles;
        dense output supports pursuit lookahead and remaining-route checks.
        """
        def clear(route):
            return PathPlanner.check_body_traversability(
                mapdata, route, body_bounds, margin=margin)

        direct = [start] if start == goal else [start, goal]
        if clear(direct):
            return PathPlanner.resample_path(direct)

        def shorten(path):
            shortened = [path[0]]
            anchor = 0
            while anchor < len(path) - 1:
                for candidate in range(len(path) - 1, anchor, -1):
                    # Include the previous segment so a shortcut cannot
                    # introduce a turn whose swept body hits an obstacle.
                    if clear(shortened[-2:] + [path[candidate]]):
                        shortened.append(path[candidate])
                        anchor = candidate
                        break
                else:
                    # A greedy shortcut may leave too little turning room.
                    return path if clear(path) else None
            return shortened

        # Distance-only grid routes can hug corners too tightly for the body.
        # If necessary, obtain a route with more clearance, then remove its
        # unnecessary bends using the same full-footprint checks.
        for use_cost_map in (False, True):
            path, _, _, _ = planner.a_star(
                start, goal, use_cost_map=use_cost_map)
            if path:
                result = shorten(path)
                if result:
                    return PathPlanner.resample_path(result)

        # A point-grid route may fit at the robot's sides while its corners
        # cannot turn. Try a graph with clearance for every body heading,
        # including map edges, before declaring a physical route unavailable.
        grid = _map_array(mapdata)
        xmin, xmax, ymin, ymax = body_bounds
        radius = math.hypot(max(abs(xmin), abs(xmax)) + margin,
                            max(abs(ymin), abs(ymax)) + margin)
        resolution = mapdata.info.resolution
        free = np.pad((grid < 50).astype(np.uint8), 1)
        clearance = cv2.distanceTransform(
            free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1] * resolution
        reserve = radius + (math.sqrt(0.5) + 0.5) * resolution
        safe = (grid >= 0) & (grid < 50) & (clearance > reserve)
        body_planner = PathPlanner.prepare(
            mapdata, np.zeros_like(grid), np.where(safe, 0, 100))
        path, _, _, _ = body_planner.a_star(start, goal)
        if path:
            result = shorten(path)
            if result:
                return PathPlanner.resample_path(result)
        return None

    @staticmethod
    def neighbors(mapdata, p, directions, must_be_walkable=True):
        neighbors = []
        for dx, dy in directions:
            candidate = (p[0] + dx, p[1] + dy)
            if not PathPlanner.is_cell_in_bounds(mapdata, candidate):
                continue
            if must_be_walkable:
                if not PathPlanner.is_cell_walkable(mapdata, candidate):
                    continue
                if dx and dy and not (
                        PathPlanner.is_cell_walkable(
                            mapdata, (p[0] + dx, p[1]))
                        and PathPlanner.is_cell_walkable(
                            mapdata, (p[0], p[1] + dy))):
                    continue
            neighbors.append(candidate)
        return neighbors

    @staticmethod
    def neighbors_of_4(mapdata, p, must_be_walkable=True):
        return PathPlanner.neighbors(
            mapdata, p, DIRECTIONS_OF_4, must_be_walkable)

    @staticmethod
    def neighbors_of_8(mapdata, p, must_be_walkable=True):
        return PathPlanner.neighbors(
            mapdata, p, DIRECTIONS_OF_8, must_be_walkable)

    @staticmethod
    def neighbors_and_distances_of_8(mapdata, p, must_be_walkable=True):
        return [(cell, _SQRT2 if cell[0] != p[0] and cell[1] != p[1] else 1.0)
                for cell in PathPlanner.neighbors_of_8(
                    mapdata, p, must_be_walkable)]

    @staticmethod
    def calc_cspace_array(mapdata: OccupancyGrid, kernel_size=21, grid=None):
        """Return C-space without serializing a ROS grid message."""
        grid = _map_array(mapdata, grid)
        kernel = _inflation_kernel(kernel_size)
        unknown = np.asarray(grid == -1, dtype=np.uint8) * 255
        unknown = cv2.erode(unknown, kernel, iterations=1)
        obstacles = (grid >= 50).astype(np.uint8)
        obstacles = cv2.dilate(obstacles, kernel, iterations=1)
        return ((obstacles != 0) | (unknown != 0)).astype(np.int8) * 100

    @staticmethod
    def calc_cspace(mapdata: OccupancyGrid, kernel_size: int = 21):
        cspace = OccupancyGrid()
        cspace.header = mapdata.header
        cspace.info = mapdata.info
        grid = PathPlanner.calc_cspace_array(mapdata, kernel_size)
        cspace.data = grid.ravel().tolist()
        return cspace

    @staticmethod
    def calc_cost_map(mapdata: OccupancyGrid, grid=None) -> np.ndarray:
        grid = _map_array(mapdata, grid)
        free_mask = np.asarray((grid >= 0) & (grid < 50), dtype=np.uint8) * 255
        dist = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)
        ratio = np.clip((20.0 - dist) / 20.0, 0.0, 1.0)
        return (ratio * ratio * 100).astype(np.uint8)

    @staticmethod
    def prepare(mapdata, cost_map, grid=None):
        """Prepare reusable search data for an occupancy grid and cost map."""
        return PlanningGrid(mapdata, cost_map, grid)

    @staticmethod
    def a_star(mapdata, cost_map, start, goal):
        return PlanningGrid(mapdata, cost_map).a_star(start, goal)

    @staticmethod
    def path_to_message(mapdata: OccupancyGrid, path, timestamp) -> Path:
        path_msg = Path()
        path_msg.header.frame_id = mapdata.header.frame_id or 'odom'
        path_msg.header.stamp = timestamp
        path_msg.poses = PathPlanner.path_to_poses(mapdata, path)
        for pose in path_msg.poses:
            pose.header.stamp = timestamp
        return path_msg
