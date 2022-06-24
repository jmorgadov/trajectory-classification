from typing import List, Tuple

import numpy as np

import feature_est as fe
from data_handler import get_selected_data

classes = {'walk': 0,
           'car': 1,
           'train': 2,
           'bus': 3,
           'bike': 4}

feat_name = [
    'distance',
    'mean_velocity',
    'median_velocity',
    'min_velocity',
    'max_velocity',
    'std_velocity',
    'var_velocity',
    'coef_var_velocity',
    'iqr_velocity',
    'velocity_change_rate',
    'stop_rate',
    'mean_acceleration',
    'median_acceleration',
    'min_acceleration',
    'max_acceleration',
    'std_acceleration',
    'var_acceleration',
    'coef_var_acceleration',
    'iqr_acceleration',
    'mean_acc_change_rate',
    'median_acc_change_rate',
    'min_acc_change_rate',
    'max_acc_change_rate',
    'std_acc_change_rate',
    'var_acc_change_rate',
    'coef_var_acc_change_rate',
    'iqr_acc_change_rate',
    'mean_angle',
    'median_angle',
    'min_angle',
    'max_angle',
    'std_angle',
    'var_angle',
    'coef_var_angle',
    'iqr_angle',
    'mean_turning_angle',
    'median_turning_angle',
    'min_turning_angle',
    'max_turning_angle',
    'std_turning_angle',
    'var_turning_angle',
    'coef_var_turning_angle',
    'iqr_turning_angle',
    'mean_heading_change_rate',
    'median_heading_change_rate',
    'min_heading_change_rate',
    'max_heading_change_rate',
    'std_heading_change_rate',
    'var_heading_change_rate',
    'coef_var_heading_change_rate',
    'iqr_heading_change_rate',
]

def convert_traj_into_vector(traj: np.ndarray, threashold: float) -> np.ndarray:
    vel = fe.velocity(traj)
    acc = fe.acceleration(traj)
    acc_chg_rate = fe.acceleration_change_rate(traj)
    ang = fe.angle(traj)
    trng_ang = fe.turning_angle(traj)
    hding_chg_rate = fe.heading_change_rate(traj)
    traj_vect = np.array(
        [
            # Distance
            fe.distance(traj),
            # Velocity
            fe.mean(vel),
            fe.median(vel),
            fe.min_val(vel),
            fe.max_val(vel),
            fe.standard_dev(vel),
            fe.variance(vel),
            fe.coef_var(vel),
            fe.iqr(vel),
            # Velocity change rate
            fe.vel_change_rate(traj, threashold),
            # Stop rate
            fe.stop_rate(traj, threashold),
            # Acceleration
            fe.mean(acc),
            fe.median(acc),
            fe.min_val(acc),
            fe.max_val(acc),
            fe.standard_dev(acc),
            fe.variance(acc),
            fe.coef_var(acc),
            fe.iqr(acc),
            # Acceleration change rate
            fe.mean(acc_chg_rate),
            fe.median(acc_chg_rate),
            fe.min_val(acc_chg_rate),
            fe.max_val(acc_chg_rate),
            fe.standard_dev(acc_chg_rate),
            fe.variance(acc_chg_rate),
            fe.coef_var(acc_chg_rate),
            fe.iqr(acc_chg_rate),
            # Angle
            fe.mean(ang),
            fe.median(ang),
            fe.min_val(ang),
            fe.max_val(ang),
            fe.standard_dev(ang),
            fe.variance(ang),
            fe.coef_var(ang),
            fe.iqr(ang),
            # Turning angle
            fe.mean(trng_ang),
            fe.median(trng_ang),
            fe.min_val(trng_ang),
            fe.max_val(trng_ang),
            fe.standard_dev(trng_ang),
            fe.variance(trng_ang),
            fe.coef_var(trng_ang),
            fe.iqr(trng_ang),
            # Heading change rate
            fe.mean(hding_chg_rate),
            fe.median(hding_chg_rate),
            fe.min_val(hding_chg_rate),
            fe.max_val(hding_chg_rate),
            fe.standard_dev(hding_chg_rate),
            fe.variance(hding_chg_rate),
            fe.coef_var(hding_chg_rate),
            fe.iqr(hding_chg_rate),
        ]
    )

    return traj_vect


def get_feat_vectors(data: List[dict]) -> Tuple[list, list, list]:
    """
    Get the list of feature vectors and their classes.

    Parameters
    ----------
    data : List[dict]
        The list of trajectories.

    Returns
    -------
    Tuple[list, list, list]
        The Vectors and the list of classes and class masks.
    """
    vectors = []
    clss_mask = []
    clss = []
    length = len(data)
    classes = {n:i for i, n in enumerate(set(md["class"] for md in data))}
    for i,d in enumerate(data):
        print(f"{(i+1)/length:.2%}", end="\r")
        traj = d["traj_data"]
        traj_vect = convert_traj_into_vector(traj, 1)
        vectors.append(traj_vect)
        clss_mask.append(class_mask(d))
        clss.append(classes[d['class']])
        
    return (vectors, clss_mask, clss)

def class_mask(traj: dict) -> np.ndarray:
    mask = np.zeros(5)
    mask[classes[traj['class']]] = 1
    return mask
