"""Frontier planning and collision checks for the differential-drive robot."""

import math
from dataclasses import dataclass
from heapq import heappop, heappush

import cv2

import numpy as np


@dataclass(frozen=True)
class Grid:
    """An axis-aligned ROS occupancy grid in odom coordinates."""

    values: np.ndarray
    origin: tuple
    resolution: float

    def cells(self, points):
        """Convert world positions to integer (x, y) grid indices."""
        return np.floor((points - self.origin) / self.resolution).astype(int)

    def centers(self, cells):
        """Convert integer grid indices to world positions at cell centers."""
        return (cells + 0.5) * self.resolution + self.origin

    def contains(self, cells):
        """Return a mask of cells inside the grid."""
        return np.all((cells >= 0) & (cells < self.values.shape[::-1]), axis=-1)


@dataclass
class FrontierMemory:
    """Keep the active neighborhood across goals and temporary stops."""

    area: Grid | None = None
    goal: tuple | None = None


_CARDINAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL_STEPS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
# Circumscribed footprint from the robot's 0.40 m by 0.30 m chassis.
_ROBOT_RADIUS = math.hypot(0.20, 0.15)
# Extra clearance prevents the lookahead/braking check from rejecting a route
# that is technically valid for the static robot footprint.
_NAVIGATION_CLEARANCE = _ROBOT_RADIUS + 0.15
_FRONTIER_CLEARANCE = _NAVIGATION_CLEARANCE
# Frontier "locales": candidate frontier cells closer than 2 * _LOCALE_RADIUS
# are one neighborhood that is worth clearing in a single visit.
_LOCALE_RADIUS = 1.0
# Metres of extra travel the planner will accept per square metre of unmapped
# space inside a locale. Raise it to favor rich locales over short tours.
_LOCALE_VALUE_WEIGHT = 0.25
# Metres of preference for staying with the locale of the previous goal. Another
# locale must beat it by this margin before the robot abandons what it started.
_LOCALE_COMMIT_BONUS = 3.0


def _route(free, start, goal):
    """Find a shortest grid path without cutting diagonally across corners."""
    height, width = free.shape
    queue = [(0.0, 0.0, start)]
    costs = {start: 0.0}
    parents = {}
    while queue:
        _, cost, cell = heappop(queue)
        if cell == goal:
            path = [cell]
            while cell != start:
                cell = parents[cell]
                path.append(cell)
            return np.array(path[::-1])
        if cost > costs[cell]:
            continue
        x, y = cell
        for dx, dy in _CARDINAL_STEPS + _DIAGONAL_STEPS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height and free[ny, nx]):
                continue
            if dx and dy and not (free[y, nx] and free[ny, x]):
                continue
            neighbor = nx, ny
            next_cost = cost + (math.sqrt(2.0) if dx and dy else 1.0)
            if next_cost < costs.get(neighbor, math.inf):
                costs[neighbor] = next_cost
                parents[neighbor] = cell
                heuristic = math.hypot(goal[0] - nx, goal[1] - ny)
                heappush(queue, (next_cost + heuristic, next_cost, neighbor))
    return None


def _travel_distances(free, start, goals, resolution):
    """Measure exact travel to candidates, stopping once they are settled."""
    _, width = free.shape
    stride = width + 2
    # Padding removes per-neighbor bounds checks; Python lists avoid NumPy
    # scalar indexing in the inner loop. Motion and corner rules match _route.
    walk = np.pad(free, 1).ravel().tolist()
    distance = [math.inf] * len(walk)
    first = int((start[1] + 1) * stride + start[0] + 1)
    targets = ((goals[:, 1] + 1) * stride + goals[:, 0] + 1).tolist()
    pending = [False] * len(walk)
    remaining = 0
    for cell in targets:
        if not pending[cell]:
            pending[cell] = True
            remaining += 1
    distance[first] = 0.0
    queue = [(0.0, first)]
    diagonal = math.sqrt(2.0)
    steps = (
        (1, 1.0, 0, 0), (-1, 1.0, 0, 0),
        (stride, 1.0, 0, 0), (-stride, 1.0, 0, 0),
        (stride + 1, diagonal, stride, 1),
        (stride - 1, diagonal, stride, -1),
        (-stride + 1, diagonal, -stride, 1),
        (-stride - 1, diagonal, -stride, -1),
    )
    while queue and remaining:
        cost, cell = heappop(queue)
        if cost > distance[cell]:
            continue
        if pending[cell]:
            pending[cell] = False
            remaining -= 1
            if not remaining:
                break
        for offset, length, side_a, side_b in steps:
            neighbor = cell + offset
            if not walk[neighbor]:
                continue
            if side_a and not (walk[cell + side_a] and walk[cell + side_b]):
                continue
            next_cost = cost + length
            if next_cost < distance[neighbor]:
                distance[neighbor] = next_cost
                heappush(queue, (next_cost, neighbor))
    return np.asarray([distance[cell] for cell in targets]) * resolution


def _remembered_area(grid, memory):
    """Reproject the active area when the map origin or resolution changes."""
    area = np.zeros(grid.values.shape, dtype=bool)
    if memory is None or memory.area is None:
        return area
    previous = memory.area
    if (previous.values.shape == grid.values.shape
            and previous.origin == grid.origin
            and previous.resolution == grid.resolution):
        return previous.values.astype(bool, copy=True)
    y, x = np.nonzero(previous.values)
    cells = grid.cells(previous.centers(np.column_stack((x, y))))
    x, y = cells[grid.contains(cells)].T
    area[y, x] = True
    return area


def _frontier_locales(grid, eligible_frontier):
    """Label frontier neighborhoods independently of their travel ranking."""
    radius = math.ceil(_LOCALE_RADIUS / grid.resolution)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    _, labels = cv2.connectedComponents(
        cv2.dilate(eligible_frontier, kernel), connectivity=8)
    return labels


def _locale_rank(grid, labels, goals, distance, unknown, previous_goal):
    """
    Rank the neighborhoods ("locales") of the candidate frontier cells.

    Frontier cells within 2 * _LOCALE_RADIUS of each other form one locale: a
    place the robot can clear in a single visit. Each locale is scored by the
    estimated length of the exploration tour that begins there (drive to the
    locale, then keep going to the nearest locale not yet visited), less a
    bonus for the unmapped area it still holds. Beginning with a distant
    locale wins whenever doing the near ones first would force a long trip
    back to it later, and the near ones are then picked up on the way back.

    Everything is rebuilt from the current map on every call. A frontier that
    appears while the robot is driving joins a nearby locale (raising its
    value) or becomes a locale of its own, and is weighed in the next plan.

    Return each goal's locale rank (0 = first).
    """
    # Dilated blobs are disjoint, so no unknown cell counts for two locales.
    unknown_area = np.bincount(
        labels.ravel(), weights=unknown.ravel()) * grid.resolution ** 2

    ids, member = np.unique(labels[goals[:, 1], goals[:, 0]],
                            return_inverse=True)
    n = len(ids)
    points = grid.centers(goals)
    size = np.bincount(member)
    centroids = np.column_stack(
        [np.bincount(member, weights=points[:, axis]) / size
         for axis in (0, 1)])
    # Cost to reach each locale from the robot: its nearest candidate cell.
    entry = np.full(n, np.inf)
    np.minimum.at(entry, member, distance)

    # Tour length for every possible first locale, finishing greedily.
    between = np.linalg.norm(centroids[:, None] - centroids[None], axis=2)
    rows = np.arange(n)
    tour = entry.copy()
    visited = np.eye(n, dtype=bool)
    here = rows
    for _ in range(n - 1):
        leg = np.where(visited, np.inf, between[here])
        nxt = leg.argmin(axis=1)
        tour += leg[rows, nxt]
        visited[rows, nxt] = True
        here = nxt

    committed = np.zeros(n, dtype=bool)
    if previous_goal is not None:
        cell = grid.cells(np.asarray(previous_goal))
        if grid.contains(cell):
            committed = ids == labels[cell[1], cell[0]]
    score = (tour - _LOCALE_VALUE_WEIGHT * unknown_area[ids]
             - _LOCALE_COMMIT_BONUS * committed)
    rank = np.empty(n, dtype=int)
    rank[np.argsort(score)] = rows
    return rank[member]


def frontier_is_productive(grid, goal):
    """Return whether a goal still borders enough unmapped space to explore."""
    cell = grid.cells(np.asarray(goal))
    if not grid.contains(cell) or grid.values[cell[1], cell[0]] != 0:
        return False
    unknown = (grid.values == -1).astype(np.float32)
    diameter = 2 * math.ceil(0.25 / grid.resolution) + 1
    gain = cv2.boxFilter(unknown, -1, (diameter, diameter),
                         borderType=cv2.BORDER_CONSTANT)
    nearby_unknown = cv2.boxFilter(unknown, -1, (3, 3),
                                   borderType=cv2.BORDER_CONSTANT)
    y, x = cell[1], cell[0]
    return bool(nearby_unknown[y, x] > 0 and gain[y, x] >= 0.1)


def frontier_points(grid, pose, hits):
    """Return all reachable, safely clear frontier centers."""
    start = grid.cells(np.asarray(pose[:2]))
    if not grid.contains(start):
        return np.empty((0, 2))
    obstacles = grid.values > 0
    if hits is not None and len(hits) > 0:
        hit_dists = np.linalg.norm(hits - pose[:2], axis=1)
        local_hits = hits[(hit_dists <= 2.0) & (hit_dists >= 0.15)]
        cells = grid.cells(local_hits)
        x, y = cells[grid.contains(cells)].T
        obstacles[y, x] = True
    distance_to_obstacle = cv2.distanceTransform(
        np.pad(~obstacles, 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1] * grid.resolution
    fits = (distance_to_obstacle
            > _NAVIGATION_CLEARANCE + grid.resolution / math.sqrt(2))
    free = ((grid.values == 0) & fits).astype(np.uint8)
    start_cell = tuple(start)
    if not free[start_cell[1], start_cell[0]]:
        free_y, free_x = np.nonzero(free)
        if len(free_x) == 0:
            return np.empty((0, 2))
        dists_sq = (free_x - start[0]) ** 2 + (free_y - start[1]) ** 2
        nearest_idx = np.argmin(dists_sq)
        if dists_sq[nearest_idx] > math.ceil(0.6 / grid.resolution) ** 2:
            return np.empty((0, 2))
        start_cell = (int(free_x[nearest_idx]), int(free_y[nearest_idx]))
    _, labels = cv2.connectedComponents(free, connectivity=4)
    reachable = labels == labels[start_cell[1], start_cell[0]]
    unknown = (grid.values == -1).astype(np.float32)
    diameter = 2 * math.ceil(0.25 / grid.resolution) + 1
    gain = cv2.boxFilter(unknown, -1, (diameter, diameter),
                         borderType=cv2.BORDER_CONSTANT)
    nearby_unknown = cv2.boxFilter(unknown, -1, (3, 3),
                                   borderType=cv2.BORDER_CONSTANT)
    frontier = (nearby_unknown > 0) & (gain >= 0.1)
    y, x = np.nonzero(frontier & reachable)
    if len(x) == 0:
        return np.empty((0, 2))
    goals = np.column_stack((x, y))
    distance = np.linalg.norm(grid.centers(goals) - pose[:2], axis=1)
    valid = ((distance > 0.20)
             & (distance_to_obstacle[y, x] >= _FRONTIER_CLEARANCE))
    return grid.centers(goals[valid])


def plan_frontier(grid, pose, hits, previous_goal, target_goal=None, memory=None):
    """
    Plan through mapped free space to a frontier or a specified goal.

    Rebuild candidates from the live map. When memory is supplied, finish the
    active neighborhood before choosing another, preserving all its remaining
    parts if a frontier splits. Within that area, prefer short actual routes
    so a larger frontier across an obstacle does not cause repeated detours.
    """
    start = grid.cells(np.asarray(pose[:2]))
    if not grid.contains(start):
        return None

    # Test candidate robot centers against raw obstacles and current returns.
    # Each occupied cell represents a square, so account for its half-diagonal.
    obstacles = grid.values > 0
    if hits is not None and len(hits) > 0:
        hit_dists = np.linalg.norm(hits - pose[:2], axis=1)
        local_hits = hits[(hit_dists <= 2.0) & (hit_dists >= 0.15)]
        cells = grid.cells(local_hits)
        x, y = cells[grid.contains(cells)].T
        obstacles[y, x] = True
    distance_to_obstacle = cv2.distanceTransform(
        np.pad(~obstacles, 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1] * grid.resolution
    fits = (distance_to_obstacle
            > _NAVIGATION_CLEARANCE + grid.resolution / math.sqrt(2))
    free = ((grid.values == 0) & fits).astype(np.uint8)

    start_cell = tuple(start)
    if not (0 <= start[1] < free.shape[0] and 0 <= start[0] < free.shape[1]
            and free[start[1], start[0]]):
        # When robot is slightly within inflation margin, snap to closest free cell.
        free_y, free_x = np.nonzero(free)
        if len(free_x) == 0:
            return None
        dists_sq = (free_x - start[0]) ** 2 + (free_y - start[1]) ** 2
        nearest_idx = np.argmin(dists_sq)
        max_search_cells = math.ceil(0.6 / grid.resolution)
        if dists_sq[nearest_idx] > max_search_cells ** 2:
            return None
        start_cell = (int(free_x[nearest_idx]), int(free_y[nearest_idx]))

    _, labels = cv2.connectedComponents(free, connectivity=4)
    reachable = labels == labels[start_cell[1], start_cell[0]]

    if target_goal is None:
        if previous_goal is None and memory is not None:
            previous_goal = memory.goal
        unknown = (grid.values == -1).astype(np.float32)
        diameter = 2 * math.ceil(0.25 / grid.resolution) + 1
        gain = cv2.boxFilter(unknown, -1, (diameter, diameter),
                             borderType=cv2.BORDER_CONSTANT)
        nearby_unknown = cv2.boxFilter(unknown, -1, (3, 3),
                                       borderType=cv2.BORDER_CONSTANT)
        frontier = (nearby_unknown > 0) & (gain >= 0.1)
        y, x = np.nonzero(frontier & reachable)
        if len(x) == 0:
            return None
        goals = np.column_stack((x, y))
        offsets = grid.centers(goals) - pose[:2]
        distance = np.linalg.norm(offsets, axis=1)
        far_enough = distance > 0.20
        if not np.any(far_enough):
            return None
        goals, offsets, distance = (goals[far_enough], offsets[far_enough],
                                    distance[far_enough])
        heading = np.array((math.cos(pose[2]), math.sin(pose[2])))
        clearance = distance_to_obstacle[goals[:, 1], goals[:, 0]]
        safely_clear = clearance >= _FRONTIER_CLEARANCE
        if not np.any(safely_clear):
            return None
        goals, offsets, distance, clearance = (
            goals[safely_clear], offsets[safely_clear],
            distance[safely_clear], clearance[safely_clear])
        obstacle_penalty = 1.5 / np.maximum(
            clearance - _ROBOT_RADIUS, grid.resolution)
        score = (distance + obstacle_penalty
                 + 0.35 * (1.0 - offsets @ heading / distance))
        keep_goal = np.zeros(len(goals), dtype=bool)
        if previous_goal is not None:
            same_frontier = (np.linalg.norm(
                grid.centers(goals) - previous_goal, axis=1) < 0.25)
            score[same_frontier] -= 1.0
            keep_goal = same_frontier
        # Group adjacent frontier cells into regions, using their remaining
        # unknown space to break ties between equally short routes.
        eligible_frontier = np.zeros_like(frontier, dtype=np.uint8)
        eligible_frontier[goals[:, 1], goals[:, 0]] = 1
        _, regions = cv2.connectedComponents(eligible_frontier,
                                             connectivity=8)
        region_ids = regions[goals[:, 1], goals[:, 0]]
        region_sizes = np.bincount(regions.ravel())
        region_gain = np.bincount(regions.ravel(), weights=gain.ravel())
        region_priority = (
            region_gain[region_ids]
            + 0.05 * region_sizes[region_ids])
        locales = _frontier_locales(grid, eligible_frontier)
        remembered = _remembered_area(grid, memory)
        active_ids = np.unique(locales[remembered & (locales != 0)])
        goal_locales = locales[goals[:, 1], goals[:, 0]]
        active = np.isin(goal_locales, active_ids)
        # Every surviving part of the active area outranks other locales.
        # Keep a productive goal, then finish the nearest remaining work;
        # productivity breaks travel ties instead of triggering long detours.
        if np.any(active):
            # No global tour is needed while committed. Search only far enough
            # to settle the active candidates, rather than the whole free map.
            travel = np.full(len(goals), np.inf)
            travel[active] = _travel_distances(
                free, start_cell, goals[active], grid.resolution)
            locale_priority = (~active).astype(int)
        else:
            travel = _travel_distances(
                free, start_cell, goals, grid.resolution)
            locale_priority = _locale_rank(
                grid, locales, goals, travel, unknown, previous_goal)
        candidate_order = np.lexsort(
            (score, -region_priority, travel,
             (~keep_goal).astype(int), locale_priority))
    else:
        target = grid.cells(np.asarray(target_goal))
        if (not grid.contains(target)
                or not free[target[1], target[0]]
                or not reachable[target[1], target[0]]):
            return None
        goals = np.asarray([target])
        candidate_order = (0,)

    path = None
    for candidate in candidate_order:
        goal = tuple(goals[candidate])
        path = _route(free, start_cell, goal)
        if path is not None:
            break
    if path is None:
        return None
    path_centers = grid.centers(path)
    if start_cell != tuple(start):
        path_centers = np.vstack(([pose[:2]], path_centers))
    if memory is not None and target_goal is None:
        if active[candidate]:
            area = remembered | np.isin(locales, active_ids)
        else:
            area = locales == goal_locales[candidate]
        memory.area = Grid(area, grid.origin, grid.resolution)
        memory.goal = tuple(path_centers[-1])
    return path_centers


def motion_is_clear(grid, pose, hits, speed, turn):
    """Check the commanded arc and braking distance against both obstacle sources."""
    # Include sensor/command latency and braking at 0.5 m/s².
    stop_time = 0.40 + speed / 0.5
    samples = max(2, math.ceil(speed * stop_time / (grid.resolution / 2)) + 1)
    t = np.linspace(0.0, stop_time, samples)
    x, y, yaw = pose
    if abs(turn) < 1e-6:
        centers = np.column_stack((x + speed * t * math.cos(yaw),
                                   y + speed * t * math.sin(yaw)))
    else:
        radius = speed / turn
        heading = yaw + turn * t
        centers = np.column_stack((x + radius * (np.sin(heading) - math.sin(yaw)),
                                   y + radius * (math.cos(yaw) - np.cos(heading))))
    # A marginally conservative current footprint must not prevent a command
    # that moves away from the return that caused the overlap. Keep the initial
    # sample for bounds and occupancy checks, but test obstacle clearance over
    # the commanded trajectory after the first control interval.
    clearance_centers = centers[1:] if speed > 0.0 else centers
    lower = grid.cells(centers.min(axis=0) - _ROBOT_RADIUS)
    upper = grid.cells(centers.max(axis=0) + _ROBOT_RADIUS)
    if not np.all(grid.contains(np.array((lower, upper)))):
        return False
    cells = grid.cells(centers)
    if np.any(grid.values[cells[:, 1], cells[:, 0]] != 0):
        return False

    # Measure to the nearest point on each occupied cell's square.
    region = grid.values[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
    y, x = np.nonzero(region > 0)
    occupied_centers = grid.centers(np.column_stack((x, y)) + lower)
    if len(occupied_centers) > 0:
        offsets = (np.abs(clearance_centers[:, None, :] - occupied_centers)
                   - grid.resolution / 2)
        distances = np.sqrt(np.sum(np.maximum(offsets, 0.0) ** 2, axis=2))
        initial_distance = distances[0]
        if np.any(np.where(initial_distance < _ROBOT_RADIUS,
                           distances < initial_distance - 1e-6,
                           distances <= _ROBOT_RADIUS)):
            return False
    if hits is not None and len(hits) > 0:
        hit_dists = np.linalg.norm(hits - pose[:2], axis=1)
        valid_hits = hits[hit_dists >= 0.15]
        if len(valid_hits) > 0:
            distances = np.linalg.norm(
                clearance_centers[:, None, :] - valid_hits, axis=2)
            initial_distance = distances[0]
            if np.any(np.where(initial_distance < _ROBOT_RADIUS,
                               distances < initial_distance - 1e-6,
                               distances <= _ROBOT_RADIUS)):
                return False
    return True
