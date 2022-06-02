"""
Parses the data, saves the trajectories and creates a metadata json file.

A trajectory is a Nx3 matrix, where N is the number of points and the 3
columns are the lat, lon and time (in seconds).

The json metadata file contains a list of dictionaries (the metada of
each trajectory) with the following keys:
    - id: the trajectory id
    - file_path: the path to the trajectory file
    - class: the class of the trajectory
"""
