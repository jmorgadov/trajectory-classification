import json
import logging
from pathlib import Path
from typing import List, NamedTuple, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

def load_trajs(metadata_file: Path) -> List[dict]:
    metadata = []
    trajs = []
    with open(metadata_file, 'r') as md:
        logging.info("Loading metadata")
        metadata = json.loads(md.read())
    for i, traj_md in enumerate(metadata):
        logging.info(
            "Processing trajectory: %s - %s", traj_md['id'], f"{(i + 1) / len(metadata):.2%}"
        )
        trajs.append({'id': traj_md['id'],
                      'traj_data': np.loadtxt(traj_md['file_path']),
                      'class': traj_md['class']})
    return trajs

def main():
    metadata_file = Path("trajectories/metadata.json")
    trajs = load_trajs(metadata_file)
    logging.info("Done")

if __name__ == "__main__":
    main()