"""
Contains all the functions to estimate the features of a given trajectory.
"""

import numpy as np
from scipy import stats as st

def distance(traj: np.ndarray):
    return np.sum(np.linalg.norm(np.diff(traj[:,:2], axis=0), axis=0))


def velocity(traj: np.ndarray):
    return np.linalg.norm(np.diff(traj[:,:2], axis=0), axis=0) / np.diff(traj[:, 2], axis=0)


def _velocity_rate(traj : np.ndarray):
    vel = velocity(traj)
    return np.linalg.norm(vel[1:] - vel[:-1], axis=0) / vel[:-1]


def vel_change_rate(traj:np.ndarray, threashold: float):
    v_rate = _velocity_rate(traj)
    return np.sum(i > threashold for i in v_rate)/distance(traj)


def stop_rate(traj: np.ndarray, threashold: float):
    vel = velocity(traj)
    return np.sum(i > threashold for i in vel)/distance(traj)


def time_difference(traj):
    return np.linalg.norm(np.diff(traj[:,2], axis=0), axis=0)


def aceleration(traj: np.ndarray):
    vel = velocity(traj)
    time_diff = time_difference(traj)
    return np.array(i/j for i,j in (vel,time_diff))


def aceleration_change_rate(traj: np.ndarray):
    aceler = aceleration(traj)
    time_diff = time_difference(traj)
    return np.array(i/j for i,j in (aceler,time_diff))


def angle(traj: np.ndarray):
    dif = np.diff(traj[:,:2], axis=0)
    return np.arctan(i/j for i,j in (dif[0], dif[1]))


def turning_angle(traj: np.ndarray):
    ang = angle(traj)
    return ang[1:] - ang[:-1]


def heading_change_rate(traj: np.ndarray):
    t_angle = turning_angle(traj)
    time_dif = time_difference(traj)
    return np.array(i/j for i,j in (t_angle, time_dif))


def rate_HCR(traj: np.ndarray):
    hcr = heading_change_rate(traj)
    time_dif = time_difference(traj)
    return np.array(i/j for i,j in (hcr, time_dif))


# General features

def mode(values: np.ndarray):
    return st.mode(values)


def mean(values: np.ndarray):
    return np.mean(values)


def median(values: np.ndarray):
    return np.median(values)


def min(values: np.ndarray):
    return np.min(values)


def max(values: np.ndarray):
    return np.max(values)


def standard_dev(values: np.ndarray):
    return np.std(values)


def variance(values: np.ndarray):
    return np.var(values)


def coef_var(values: np.ndarray):
    return standard_dev(values)/abs(mean(values))


def percentile(values: np.ndarray, percent_value: int):
    return np.percentile(values, percent_value)


def iqr(values: np.ndarray):
    return percentile(values, 75) - percentile(values, 25)
