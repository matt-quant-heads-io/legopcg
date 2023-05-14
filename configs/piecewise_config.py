import os
import pathlib
import datetime

PIECEWISE_CONFIG = {
    "data": {
 
    },
    "train" : {
        "num_episodes" : 10000,
        "policy" : "MultiInputPolicy",#"MlpPolicy", #"CnnPolicy"
        "GridDimensions" : [10,10,10],
        "LegoBlockIDs" : ["empty", "3005", "3004", "3622"],
        "LegoBlockDimensions" : {
            "empty" : [0,0,0],
            "3005" : [1,1,1], 
            "3004" : [2,1,1],
            "3622" : [3,1,1],
            "3003" : [2,1,2],
            "11212" : [3,1,3],
        },
        "num_of_blocks" : 20,#20,
        "observation_size" : 21, #try 7
        "reward_param": "avg_height",
        "reps_per_episode": 6, #try 10, 5, 20, 25, 30
        "scheduled_episodes": False,
        "punish": True
    },
    
    "model" :{
        #"log_path" : f"{os.getcwd()}/logs",
        "saved_model_path" : f"{os.getcwd()}/saved_models/",
        "log_path" : f"{os.getcwd()}/logs/",
        "model_name" : "lego_agent"
    },
}
