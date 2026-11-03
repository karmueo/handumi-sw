"""Replay playback starts at the robot home pose (like sim teleop)."""

from __future__ import annotations

import numpy as np

from handumi.scripts.replay import replay_in_sim as replay
from handumi.scripts.replay.replay_in_sim import joint_approach_ramp

HOME = np.array([0.0, -0.9, 0.6, 0.0], dtype=np.float32)
START = np.array([-0.22, -0.78, 0.18, 0.02], dtype=np.float32)


def test_ramp_starts_at_home_and_stops_before_the_episode() -> None:
    ramp = joint_approach_ramp(HOME, START, frames=30)
    assert ramp.shape == (30, HOME.size)
    np.testing.assert_allclose(ramp[0], HOME, atol=1e-6)
    # The solved trajectory supplies START itself, so the ramp must not repeat it.
    assert float(np.abs(ramp[-1] - START).max()) > 1e-4
    assert float(np.abs(ramp[-1] - START).max()) < float(np.abs(HOME - START).max())


def test_ramp_is_monotonic_and_smooth() -> None:
    ramp = joint_approach_ramp(HOME, START, frames=30)
    steps = np.diff(np.concatenate([ramp, START[None]]), axis=0)
    direction = np.sign(START - HOME)
    assert np.all(np.sign(steps) * direction >= 0)  # never overshoots or backs up
    # Smoothstep easing: no step may exceed the naive linear step by 2x.
    assert float(np.abs(steps).max()) < 2.0 * float(np.abs(START - HOME).max() / 30)


def test_ramp_is_skipped_when_disabled_or_already_home() -> None:
    assert joint_approach_ramp(HOME, START, frames=0).shape == (0, HOME.size)
    assert joint_approach_ramp(HOME, HOME.copy(), frames=30).shape == (0, HOME.size)


def test_rm75_absolute_tcp_defaults_to_direct_start() -> None:
    """RM75B 绝对 TCP 默认从首帧开始，显式接近时间及其他模式仍生效。"""
    parser = replay.build_parser()
    direct = parser.parse_args(["unused", "--robot", "rm75b", "--retarget-mode", "absolute-tcp"])
    replay.resolve_replay_options(direct)
    assert direct.approach_seconds == 0.0
    assert direct.initial_position_tolerance_m == 0.02
    assert direct.max_ik_position_error_m == 0.02
    assert direct.max_ik_rotation_error_deg == 10.0
    approach = parser.parse_args(["unused", "--robot", "rm75b", "--retarget-mode", "absolute-tcp", "--approach-seconds", "1"])
    replay.resolve_replay_options(approach)
    assert approach.approach_seconds == 1.0
    other = parser.parse_args(["unused", "--robot", "rm75b", "--retarget-mode", "local-relative"])
    replay.resolve_replay_options(other)
    assert other.approach_seconds == 1.0
