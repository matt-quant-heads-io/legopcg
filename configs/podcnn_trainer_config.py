import os
import pathlib
import datetime


basepath = pathlib.Path(os.getcwd()).parents[0]
PODCNN_TRAINER_CONFIG = {
    "data": {
        "path": f"{basepath}/legopcg/data/input/podcnn/vintage_car",
        "goals_path": f"{basepath}/legopcg/data/goals/vintage_car_1.mpd",
        "output_path": f"{basepath}/legopcg/data/output",
        "action_dim": 37,
        "train_data_path": f"{basepath}/legopcg/data/trajectories/racers",
        "obs_size": 6,
        "use_signed_inputs": True,
    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,
        "epochs": 500,
        "steps_per_epoch": 128,
        "val_subsplits": 5,
        "optimizer": {"type": "adam"},
        "metrics": ["accuracy"],
    },
    "model": {
        "obs_size": 6,
        "action_dim": 37,
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3,
        },
        "output": 3,
        "model_save_path": f"{basepath}/legopcg/saved_models",
        "model_name": f"pod_cnn_{datetime.datetime.now()}.h5",
    },
}
