"""Model config in json format"""
import os
import pathlib
import datetime

POD_TRAJECTORY_GENERATOR_CONFIG = {
    "data": {
        "goals_path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/data/goals",
        "goal_files": ["dune_rover.mpd"],
        "trajectory_output_path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/data/pod_trajectories",
        "action_dim": 4
    }
}