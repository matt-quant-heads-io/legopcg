"""Model config in json format"""
import os
import pathlib
import datetime


CONDITIONAL_PODCNN_RACERS_CONFIG = {
    "data": {
        "goal_path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/data/goals/racers",
        "trajectories_path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/data/trajectories/racers",
        "action_dim": 38,
        "num_training_steps": 1000000,
        # TODO: move maps below into problem object
        "actions_map": {'recte4.dat': 0, '3023.dat': 1, '44674.dat': 2, '3021.dat': 3, '51719.dat': 4, '4081a.dat': 5, '4589.dat': 6, '50947.dat': 7, '4600.dat': 8, '51011.dat': 9, '50951.dat': 10, '2432.dat': 11, '30603.dat': 12, '30027a.dat': 13, '3020.dat': 14, '2412b.dat': 15, '30602.dat': 16, '3795.dat': 17, '3710.dat': 18, '41854.dat': 19, '30028.dat': 20, '48183.dat': 21, '3839a.dat': 22, '2412a.dat': 23, '54200.dat': 24, '3022.dat': 25, '43719.dat': 26, '3031.dat': 27, '6141.dat': 28, '6015.dat': 29, '6014.dat': 30, '2540.dat': 31, '3938.dat': 32, '3034.dat': 34, '32028.dat': 35, '3937.dat': 36},
        "rev_actions_map": {0: 'recte4.dat', 1: '3023.dat', 2: '44674.dat', 3: '3021.dat', 4: '51719.dat', 5: '4081a.dat', 6: '4589.dat', 7: '50947.dat', 8: '4600.dat', 9: '51011.dat', 10: '50951.dat', 11: '2432.dat', 12: '30603.dat', 13: '30027a.dat', 14: '3020.dat', 15: '2412b.dat', 16: '30602.dat', 17: '3795.dat', 18: '3710.dat', 19: '41854.dat', 20: '30028.dat', 21: '48183.dat', 22: '3839a.dat', 23: '2412a.dat', 24: '54200.dat', 25: '3022.dat', 26: '43719.dat', 27: '3031.dat', 28: '6141.dat', 29: '6015.dat', 30: '6014.dat', 31: '2540.dat', 32: '3938.dat', 34: '3034.dat', 35: '32028.dat', 36: '3937.dat'}
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
        "saved_model_name": f"cpodcnn_racers_{datetime.datetime.now()}.h5"

    }
}


