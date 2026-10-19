"""Typed tool parameter schemas and the structured Observation every tool returns. No ROS imports.

Depth/position bounds are deliberately NOT encoded here: the safety validator owns limits (they are
configurable and violations must be caught and counted, not silently rejected by the parser).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Params(BaseModel):
    model_config = ConfigDict(extra='forbid')


class GetDetectionsParams(_Params):
    pass


class NavigateToParams(_Params):
    target: Optional[str] = Field(None, description='object to go to (e.g. gate, red_buoy); use this OR x/y '
                                                    'OR forward_m/left_m')
    x: Optional[float] = Field(None, description='world x in metres (forward along the pool)')
    y: Optional[float] = Field(None, description='world y in metres (left positive)')
    forward_m: Optional[float] = Field(None, description='relative move: metres ahead of the vehicle (negative = back)')
    left_m: Optional[float] = Field(None, description='relative move: metres to the vehicle\'s left (negative = right)')
    depth: Optional[float] = Field(None, description='metres below the surface; default = object depth / current depth')
    pass_through: Optional[bool] = Field(None, description='for target=gate: drive through the opening '
                                         '(default true for the gate unless standoff_m > 0)')
    standoff_m: float = Field(0.0, description='stop this far before the target (0 = touch it)')
    speed: Optional[float] = Field(None, description='m/s, optional')

    @property
    def mode(self) -> str:
        if self.target is not None:
            return 'target'
        if self.forward_m is not None or self.left_m is not None:
            return 'relative'
        return 'point'

    @property
    def passes_gate(self) -> bool:
        """Stopping inside a gate is never useful: target=gate means 'go through' unless a standoff
        asks to stop in front of it."""
        if self.target != 'gate':
            return False
        return self.pass_through if self.pass_through is not None else self.standoff_m <= 0

    @model_validator(mode='after')
    def _target_or_point(self):
        modes = [self.target is not None, self.x is not None or self.y is not None,
                 self.forward_m is not None or self.left_m is not None]
        if sum(modes) != 1:
            raise ValueError('navigate_to takes exactly one of: target, x/y/depth, forward_m/left_m')
        if modes[1] and (self.x is None or self.y is None or self.depth is None):
            raise ValueError('navigate_to with a point needs all of x, y, depth')
        if self.pass_through and self.target != 'gate':
            raise ValueError('pass_through only applies to target=gate')
        return self


class SearchPatternParams(_Params):
    target: str = Field(..., description='object class to look for')
    pattern: Literal['yaw_sweep', 'spiral', 'lawnmower'] = 'yaw_sweep'
    radius_m: float = Field(6.0, description='search extent in metres')
    depth: Optional[float] = Field(None, description='search depth in metres; default current depth')


class HoldDepthParams(_Params):
    depth: float = Field(..., description='metres below the surface')
    duration_s: float = Field(5.0, description='seconds to hold')


class GetBatteryParams(_Params):
    pass


class GetPoseParams(_Params):
    pass


class SurfaceParams(_Params):
    pass


class AbortParams(_Params):
    reason: str = ''


TOOL_PARAMS: dict[str, type[_Params]] = {
    'get_detections': GetDetectionsParams,
    'navigate_to': NavigateToParams,
    'search_pattern': SearchPatternParams,
    'hold_depth': HoldDepthParams,
    'get_battery': GetBatteryParams,
    'get_pose': GetPoseParams,
    'surface': SurfaceParams,
    'abort': AbortParams,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    'get_detections': 'List objects currently seen by the camera plus remembered object positions.',
    'navigate_to': 'Move to an object (target=..., e.g. touch a buoy with standoff_m=0, or '
                   'target=gate with pass_through=true), to a world point (x, y, depth), or relative to the '
                   'vehicle (forward_m, left_m, optional depth). '
                   'Fails with target_not_detected if the object has not been seen yet.',
    'search_pattern': 'Run a search pattern (yaw_sweep | spiral | lawnmower) until target is detected.',
    'hold_depth': 'Hold a depth in place for duration_s seconds.',
    'get_battery': 'Read battery percentage.',
    'get_pose': 'Read vehicle position (x, y, depth) and heading.',
    'surface': 'Ascend to the surface and stay there.',
    'abort': 'Stop the mission safely: cancel motion and surface.',
}

ErrorCode = Literal['target_not_detected', 'timeout', 'action_failed', 'unsafe_target',
                    'invalid_params', 'backend_unavailable', 'unknown_tool', 'not_found',
                    # v2: preempted by a newer command / by the health & safety monitor's veto
                    'canceled', 'health_veto']


class Observation(BaseModel):
    tool: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    error_code: Optional[ErrorCode] = None
    duration_s: float = 0.0          # wall seconds
    sim_time: Optional[float] = None  # sim seconds at completion

    def brief(self) -> str:
        if self.ok:
            return f'{self.tool}: ok {self.data}'
        return f'{self.tool}: FAILED [{self.error_code}] {self.error} {self.data}'
