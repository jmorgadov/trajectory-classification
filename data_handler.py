"""
This contains necessary functions to handle the data.
"""
import json
from pathlib import Path
from typing import List

import numpy as np


def load_trajs_data(metadata_file: Path) -> List[dict]:
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

    for traj_md in metadata:
        traj_md["traj_data"] = np.loadtxt(traj_md["file_path"])
    return metadata
