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
import os, numpy as np, datetime as dt, re, json
from pathlib import Path

root = Path('./geolife_dataset')
metadata = []

def trajectories_usr(usr: Path):
    if usr.joinpath('labels.txt').exists():
        labels = parse_labels(open(usr.joinpath('labels.txt'),'r').read())
        trajs = parse_an_user_traj(usr)
        try:
            os.mkdir('./trajectories/'+usr.name)
        except:
            pass
        index = 0
        i = 0
        dt0 = labels[index][0]
        dte = labels[index][1]
        traj = []
        while i < len(trajs):
            reg_date = parse_a_date(trajs[i][2])
            reg_time = parse_a_time(trajs[i][3])
            reg_datetime = create_datetime(reg_date, reg_time)
            if dt0 <= reg_datetime <= dte:
                traj.append([trajs[i][0], trajs[i][1], range_time(dt0, reg_datetime)])
            elif dte < reg_datetime:
                i-=1
                if len(traj) > 0:
                    traj = np.array(traj)
                    class_ = labels[index][2]
                    ID = usr.name +'_'+ str(index)
                    file_path = './trajectories/'+usr.name+'/'+str(index)+'_'+ class_
                    np.savetxt(file_path, traj)
                    metadata.append({'id': ID, 'file_path': file_path, 'class': class_})
                    traj = []
                index += 1
                if index >= len(labels): 
                    return 0
                dt0 = labels[index][0]
                dte = labels[index][1]
            i+=1
    return -1

def parse_an_user_traj(usr: Path) -> list:
    trajs_path = usr.joinpath('Trajectory')
    data = []
    sorted_paths = sorted([i for i in trajs_path.iterdir()])
    for plt in sorted_paths:
        with open(plt, 'r') as reg:
            data += parse_a_plt(reg.read())
    return data
    
def parse_a_plt(plt: str) -> list:
    lines = plt.splitlines()[6:]
    reg_plt = []
    for line in lines:
        reg_plt.append(parse_a_register(line))
    return reg_plt

def parse_labels(labels: str) -> list:
    clfs = []
    labels = labels.splitlines()[1:]
    for line in labels:
        clfs.append(parse_label(line))
    return clfs

def parse_label(line: str) -> list:
    line = line.split()
    ds, ts = parse_a_date(line[0]), parse_a_time(line[1])
    de, te = parse_a_date(line[2]), parse_a_time(line[3])
    sdt = create_datetime(ds, ts)
    edt = create_datetime(de, te)
    return [sdt, edt, line[4]]

def parse_a_register(register: str) -> list:
    reg = register.split(',')
    lat, lon = float(reg[0]), float(reg[1])
    date, time = reg[5], reg[6]
    return [lat, lon, date, time]
        
def range_time(dt1: dt.datetime, dt2: dt.datetime) -> int:
    time = dt2 - dt1
    return time.seconds
    
def parse_a_date(date: str) -> list:
    d = [int(i) for i in re.split('/|-', date)]
    return d

def parse_a_time(time: str) -> list:
    t = [int(i) for i in time.split(':')]
    return t
    
def create_datetime(date: list, time: list) -> dt.datetime:
    return dt.datetime(year=date[0], month=date[1], day=date[2], 
                  hour=time[0], minute=time[1], second=time[2])

if __name__ == "__main__":
    for usr in root.iterdir():
        trajectories_usr(usr)
        
    with open('./trajectories/metadata.json', "w", encoding="utf-8") as doc:
        json.dump(metadata, doc, indent=4, ensure_ascii=False)