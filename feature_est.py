"""
Contains all the functions to estimate the features of a given trajectory.
"""

import numpy as np
from numpy.linalg import norm


def delta_r(traj: np.ndarray):
    return norm(np.diff(traj[:, :2], axis=0), axis=0)


def delta_t(traj: np.ndarray):
    return np.diff(traj[:, 2], axis=0)


def distance(traj: np.ndarray):
    return np.sum(delta_r(traj))


def velocity(traj: np.ndarray):
    return delta_r(traj) / delta_t(traj)


def _velocity_rate(traj: np.ndarray):
    vel = velocity(traj)
    return norm(vel[1:] - vel[:-1], axis=0) / vel[:-1]


def vel_change_rate(traj: np.ndarray, threashold: float):
    v_rate = _velocity_rate(traj)
    return np.sum(v_rate > threashold) / distance(traj)


def stop_rate(traj: np.ndarray, threashold: float):
    vel = velocity(traj)
    return np.sum(vel < threashold) / distance(traj)


def acceleration(traj: np.ndarray):
    vel = velocity(traj)
    _delta_t = delta_t(traj)
    return np.diff(vel) / _delta_t[:-1]


def acceleration_change_rate(traj: np.ndarray):
    acc = acceleration(traj)
    _delta_t = delta_t(traj)
    return np.diff(acc) / _delta_t[:-2]


def angle(traj: np.ndarray):
    delta_y = np.diff(traj[:, 1])
    delta_x = np.diff(traj[:, 0])
    return np.arctan(delta_y / delta_x)


def turning_angle(traj: np.ndarray):
    return np.diff(angle(traj))


def heading_change_rate(traj: np.ndarray):
    t_angle = turning_angle(traj)
    _delta_t = delta_t(traj)
    return t_angle / _delta_t


def rate_HCR(traj: np.ndarray):
    hcr = heading_change_rate(traj)
    _delta_t = delta_t(traj)
    return hcr / _delta_t[:-1]


# General features


def mean(values: np.ndarray):
    return np.mean(values)


def median(values: np.ndarray):
    return np.median(values)


def min_val(values: np.ndarray):
    return np.min(values)


def max_val(values: np.ndarray):
    return np.max(values)


def standard_dev(values: np.ndarray):
    return np.std(values)


def variance(values: np.ndarray):
    return np.var(values)


def coef_var(values: np.ndarray):
    return standard_dev(values) / np.abs(mean(values))


def percentile(values: np.ndarray, percent_value: int):
    return np.percentile(values, percent_value)


def iqr(values: np.ndarray):
    return percentile(values, 75) - percentile(values, 25)
