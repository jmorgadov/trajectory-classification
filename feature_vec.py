from typing import List, Tuple

import numpy as np

import feature_est as fe
from data_handler import get_selected_data

classes = {'walk': 0,
           'car': 1,
           'train': 2,
           'bus': 3,
           'bike': 4}

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

    for d in data:
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
