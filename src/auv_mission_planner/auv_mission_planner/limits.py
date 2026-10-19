"""Safety limits shared by the plan validator (static) and the tools layer (runtime). No ROS imports."""
from __future__ import annotations

from typing import Optional


def point_violation(limits: dict, x: Optional[float], y: Optional[float], depth: Optional[float]) -> Optional[str]:
    """Return a reason string if the point breaks depth or geofence limits, else None."""
    if depth is not None:
        if depth < limits['min_depth_m']:
            return f'depth {depth:.2f} m is above the minimum {limits["min_depth_m"]} m'
        if depth > limits['max_depth_m']:
            return f'depth {depth:.2f} m exceeds the maximum {limits["max_depth_m"]} m'
    g = limits.get('geofence')
    if g and x is not None and y is not None:
        if not (g['x_min'] <= x <= g['x_max'] and g['y_min'] <= y <= g['y_max']):
            return (f'point ({x:.1f}, {y:.1f}) is outside the geofence '
                    f'x[{g["x_min"]}, {g["x_max"]}] y[{g["y_min"]}, {g["y_max"]}]')
    return None
