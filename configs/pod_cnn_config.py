"""Model config in json format"""
import os
import pathlib
import datetime

PODCNN_CONFIG = {
    "data": {
        "path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/data/input/podcnn/vintage_car",
        "image_size": 128,
        "load_with_info": True,
        "num_classes": 4,
        "dims": 3,
        "obs_size": 10,
        "move_inc": 100,
        "goals_path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/data/goals/vintage_car_1.mpd",
        "num_gen_episodes": 5000,
        "output_path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/data/output",
        "action_dim": 4
    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,
        "epochs": 500,
        "steps_per_epoch": 128,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3,
        "saved_model_path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/saved_models",
        "saved_model_name": f"pod_cnn_{datetime.datetime.now()}.h5"

    }
}



print(PODCNN_CONFIG["data"]["path"])