import os

LEGO3D_CONFIG = {
    "data":{

    },
    "train" : {
        "num_timesteps" : 30000,
        "num_envs" : 2,
        "total_bricks" : 70,
        "policy" : "MlpPolicy",
        "grid_dimensions" : [10,10,10],
        "crop_dimensions" : [5,5,5],
        # "lego_block_ids" : ["empty", "3005", "3004", "3622"],
        "lego_block_ids" : ["empty", "3005"],
        "lego_block_dims" : {
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