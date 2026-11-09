"""Pure-pursuit steering for following a planned path."""

import math

import numpy as np

from .frontier_explorer import Grid, motion_is_clear


def command_for_path(path, pose, max_speed, max_turn, grid: Grid, hits):
    """Return a safe pure-pursuit command, or None when no arc is safe."""
    distances = np.linalg.norm(path - pose[:2], axis=1)
    ahead = path[np.argmin(distances):]
    steps = np.linalg.norm(np.diff(ahead, axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(steps)))
    for lookahead in (0.35, 0.15):
        target = ahead[min(np.searchsorted(arc_length, lookahead),
                           len(ahead) - 1)]
        dx, dy = target - pose[:2]
        distance = math.hypot(dx, dy)
        heading_error = math.remainder(
            math.atan2(dy, dx) - pose[2], math.tau)
        if abs(heading_error) > 0.55:
            turn = np.clip(2.0 * heading_error, -max_turn, max_turn)
            if motion_is_clear(grid, pose, hits, 0.0, turn):
                return 0.0, float(turn)
            continue
        speed = min(max_speed, distance, distances[-1])
        curvature = 2 * math.sin(heading_error) / max(distance, 0.05)
        for factor in (1.0, 0.45):
            linear = speed * factor
            turn = np.clip(linear * curvature, -max_turn, max_turn)
            if motion_is_clear(grid, pose, hits, linear, turn):
                return float(linear), float(turn)
    return None
