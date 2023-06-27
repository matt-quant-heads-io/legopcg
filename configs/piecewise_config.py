import os
import pathlib
import datetime

PIECEWISE_CONFIG = {
    "data": {
 
    },
    "train" : {
        "num_episodes" : 6000,
        "policy" : "MultiInputPolicy",#""CnnPolicy",MlpPolicy", #
        "GridDimensions" : [15,3*15,15],
        "LegoBlockIDs" : ["empty", "3005", "3004", "3622"],
        "LegoBlockDimensions" : {
            "empty" : [0,0,0],
            "3005" : [1,3,1], 
            "3004" : [2,3,1],
            "3003" : [2,3,2],
            "3031" : [4,1,4], #1 y coordinate unit is 8 LDU
        },
        "num_of_blocks" : 20,#20,
        "observation_size" : 31, #try 7
        "reward_param": "avg_height", #"volume_covered", # #platform, 
        "reps_per_episode": 20, #try 10, 5, 20, 25, 30
        "scheduled_episodes": False,
        "punish": 0
    },
    
    "model" :{
        #"log_path" : f"{os.getcwd()}/logs",
        "saved_model_path" : f"{os.getcwd()}/saved_models/",
        "log_path" : f"{os.getcwd()}/logs/",
        "model_name" : "lego_agent",
        "features_extractor": "cnn",#"default"#, "cnn"
    },
}
