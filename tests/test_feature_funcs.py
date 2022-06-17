import numpy as np
import pytest

import feature_est as fe

# pylint: disable=W0621


@pytest.fixture
def traj():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 2.0],
            [3.0, 7.0, 4.0],
            [4.0, 7.0, 6.0],
        ]
    )


def test_delta_r(traj):
    assert fe.delta_r(traj) == pytest.approx([5.0, 3.0, 1.0])


def test_delta_t(traj):
    assert fe.delta_t(traj) == pytest.approx([2.0, 2.0, 2.0])


def test_distance(traj):
    assert fe.distance(traj) == pytest.approx(9.0)


def test_velocity(traj):
    assert fe.velocity(traj) == pytest.approx([2.5, 1.5, 0.5])


def test_velocity_rate(traj):
    assert fe._velocity_rate(traj) == pytest.approx([1 / 2.5, 1 / 1.5])


def test_vel_change_rate(traj):
    assert fe.vel_change_rate(traj, 0.5) == pytest.approx(1 / 9.0)


def test_stop_rate(traj):
    assert fe.stop_rate(traj, 2.0) == pytest.approx(2 / 9.0)


def test_acceleration(traj):
    assert fe.acceleration(traj) == pytest.approx([-0.5, -0.5])


def test_acceleration_change_rate(traj):
    assert fe.acceleration_change_rate(traj) == pytest.approx([0.0])


def test_angle(traj):
    assert fe.angle(traj) * 180 / np.pi == pytest.approx([53.13010, 90, 0])


def test_turning_angle(traj):
    assert fe.turning_angle(traj) * 180 / np.pi == pytest.approx([90 - 53.13010, -90])


def test_heading_change_rate(traj):
    assert fe.heading_change_rate(traj) * 180 / np.pi == pytest.approx(
        [(90 - 53.13010) / 2, -45]
    )


def test_rate_hcr(traj):
    assert fe.rate_hcr(traj) * 180 / np.pi == pytest.approx(
        [-45 / 2 - (90 - 53.13010) / 4]
    )
