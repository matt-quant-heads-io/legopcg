"""Model config in json format"""
import os
import pathlib
import sys

CMAMAE_CONFIG = {
    "model":{
        "archive": {
            "solution_dim": 100,
            "dims": (100, 100),
            "max_bound": 100 / 2 * 5.12,
            "learning_rate": 0.01,
            "threshold_min": 0.0
        },
        "emitters": {
            "sigma": 0.5,
            "ranker": "imp",
            "selection_rule": "mu",
            "restart_rule": "basic",
            "batch_size": 36,
            "num_emitters": 15
        }},
    "data": {
        
    },
    "train": {
        "total_iterations": 10_000,
        "iters_per_progress_update": 500
    }
}


