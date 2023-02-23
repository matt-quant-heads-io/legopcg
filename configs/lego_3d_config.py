import os

LEGO3D_CONFIG = {
    "data":{

    },
    "train" : {
        "num_timesteps" : 150000,
        "policy" : "MlpPolicy",
        "GridDimensions" : [10,10,10],
        "LegoBlockIDs" : ["empty", "3005", "3004", "3622"],
        "LegoBlockDimensions" : {
            "empty" : [0,0,0],
            "3005" : [1,1,1], 
            "3004" : [2,1,1],
            "3622" : [3,1,1]
        }
    },
    
    "model" :{
        "log_path" : f"{os.getcwd()}/logs",
        "saved_model_path" : f"{os.getcwd()}/saved_models",
        "animations_path" : f"{os.getcwd()}/animations",
        "model_name" : "lego_agent",
    },
}