"""
Contains all the functions to estimate the features of a given trajectory.
"""

import numpy as np
import numpy.typing as npt
from numpy.linalg import norm

FloatArray = npt.NDArray[np.float64]


def delta_r(traj: FloatArray) -> FloatArray:
    """
    Position difference between two consecutive points.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    FloatArray
        Position difference between two consecutive points.
    """
    return norm(np.diff(traj[:, :2], axis=0), axis=0)


def delta_t(traj: FloatArray) -> FloatArray:
    """
    Time difference between two consecutive points.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    FloatArray
        Time difference between two consecutive points.
    """
    return np.diff(traj[:, 2], axis=0)


def distance(traj: FloatArray) -> float:
    """
    Total distance traveled.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Total distance traveled.
    """

    return np.sum(delta_r(traj))


def velocity(traj: FloatArray) -> FloatArray:
    """
    Velocity of the trajectory at each point.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    FloatArray
        Velocity of the trajectory at each point.
    """
    return delta_r(traj) / delta_t(traj)


def _velocity_rate(traj: FloatArray) -> FloatArray:
    vel = velocity(traj)
    return norm(vel[1:] - vel[:-1], axis=0) / vel[:-1]


def vel_change_rate(traj: FloatArray, threashold: float) -> float:
    """
    Velocity change rate.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Velocity change rate.
    """
    v_rate = _velocity_rate(traj)
    return np.sum(v_rate > threashold) / distance(traj)


def stop_rate(traj: FloatArray, threashold: float) -> float:
    """
    Stop rate.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Stop rate.
    """
    vel = velocity(traj)
    return np.sum(vel < threashold) / distance(traj)


def acceleration(traj: FloatArray) -> FloatArray:
    """
    Acceleration of the trajectory at each point.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    FloatArray
        Acceleration of the trajectory at each point.
    """
    vel = velocity(traj)
    _delta_t = delta_t(traj)
    return np.diff(vel) / _delta_t[:-1]


def acceleration_change_rate(traj: FloatArray) -> FloatArray:
    """
    Acceleration change rate.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    FloatArray
        Acceleration change rate.
    """
    acc = acceleration(traj)
    _delta_t = delta_t(traj)
    return np.diff(acc) / _delta_t[:-2]


def angle(traj: FloatArray) -> FloatArray:
    """
    Angle of the trajectory at each point.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    FloatArray
        Angle of the trajectory at each point.
    """
    delta_y = np.diff(traj[:, 1])
    delta_x = np.diff(traj[:, 0])
    return np.arctan(delta_y / delta_x)


def turning_angle(traj: FloatArray) -> FloatArray:
    """
    Turning angle of the trajectory at each point.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    FloatArray
        Turning angle of the trajectory at each point.
    """
    return np.diff(angle(traj))


def heading_change_rate(traj: FloatArray) -> FloatArray:
    """
    Heading change rate of the trajectory at each point.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    FloatArray
        Heading change rate of the trajectory at each point.
    """
    t_angle = turning_angle(traj)
    _delta_t = delta_t(traj)
    return t_angle / _delta_t


def rate_hcr(traj: FloatArray) -> FloatArray:
    """
    Rate of change of heading change rate.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    FloatArray
        Rate of change of heading change rate.
    """
    hcr = heading_change_rate(traj)
    _delta_t = delta_t(traj)
    return hcr / _delta_t[:-1]


# General features


def mean(values: FloatArray) -> float:
    """
    Mean of the values.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Mean of the values.
    """
    return np.mean(values)


def median(values: FloatArray) -> float:
    """
    Median of the values.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Median of the values.
    """
    return values[len(values) // 2]


def min_val(values: FloatArray) -> float:
    """
    Minimum value.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Minimum value.
    """
    return np.min(values)


def max_val(values: FloatArray) -> float:
    """
    Maximum value.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Maximum value.
    """
    return np.max(values)


def standard_dev(values: FloatArray) -> float:
    """
    Standard deviation.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Standard deviation.
    """
    return np.std(values)


def variance(values: FloatArray) -> float:
    """
    Variance.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Variance.
    """
    return np.var(values)


def coef_var(values: FloatArray) -> float:
    """
    Coefficient of variation.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Coefficient of variation.
    """
    return standard_dev(values) / np.abs(mean(values))


def percentile(values: FloatArray, percent_value: float) -> float:
    """
    Percentile.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Percentile.
    """
    assert 0 <= percent_value <= 100
    return float(np.percentile(values, percent_value))


def iqr(values: FloatArray) -> float:
    """
    Interquartile range.

    Parameters
    ----------
    traj : FloatArray
        Trajectory.

    Returns
    -------
    float
        Interquartile range.
    """
    return percentile(values, 75) - percentile(values, 25)
