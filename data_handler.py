"""
This contains necessary functions to handle the data.
"""
import json
from pathlib import Path
from typing import List

import numpy as np


def load_trajs_metadata(metadata_file: Path) -> List[dict]:
    """
    Loads all the data from the metadata file.

    Parameters
    ----------
    metadata_file : Path
        Path to the metadata file.

    Returns
    -------
    List[dict]
        The data and metadata of each trajectory.
    """
    with open(metadata_file, "r", encoding="utf-8") as md_file:
        metadata: List[dict] = json.load(md_file)
    return metadata


def load_trajs_data(metadata: List[dict]) -> List[dict]:
    """
    Loads all the data from the trajectories file.

    Parameters
    ----------
    metadata : List[dict]
        The metadata of each trajectory.

    Returns
    -------
    List[dict]
        The metadata of each trajectory with the trajectory data included.
    """
    for i, traj_md in enumerate(metadata):
        print(f"{(i+1)/len(metadata):.2%}", end="\r")
        traj_md["traj_data"] = np.loadtxt(traj_md["file_path"], ndmin=2)
    return metadata


def get_selected_data() -> List[dict]:
    """
    Loads the selected data from the metadata file.

    Returns
    -------
    List[dict]
        The metadata of each trajectory with the trajectory data included.
    """
    metadata_file = Path("trajectories/metadata.json")
    metadata = load_trajs_metadata(metadata_file)
    classes = {"car", "taxi", "bus", "walk", "bike", "subway", "train"}
    data = [traj_md for traj_md in metadata if traj_md["class"] in classes]
    data = load_trajs_data(data)
    final_data = []
    for traj_md in data:
        # Filter trajs: dt <= 3s and len >= 100
        if (
            np.mean(np.diff(traj_md["traj_data"][:, 2])) > 3
            or traj_md["traj_data"].shape[0] < 100
        ):
            continue

        # Join similar classes
        if traj_md["class"] == "taxi":
            traj_md["class"] = "car"
        if traj_md["class"] == "subway":
            traj_md["class"] = "train"
        final_data.append(traj_md)
    return final_data
