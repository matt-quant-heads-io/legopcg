import os 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

# local imports
from .base_model import BaseModel
from utils import utils as ut 
from gym_pcgrl.envs.pcgrl_env_3d import LegoPCGEnv3D

class LegoModel3D(BaseModel):

    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)

        # unpack config
        self.train_config = self.config.train
        self.model_config = self.config.model
        self.policy = self.train_config["policy"]
        self.num_timesteps = self.train_config["num_timesteps"]
        self.lego_blocks_dims_dict = self.train_config["LegoBlockDimensions"]
        self.log_path = self.model_config["log_path"]
        self.saved_model_path = os.path.join(self.model_config["saved_model_path"],
                                self.model_config["model_name"])
        self.animations_path = self.model_config["animations_path"]

        self.model = None
        self.device = ut.get_device()
        self.env = DummyVecEnv([lambda: LegoPCGEnv3D()])

        # train or load a trained model
        if mode == "train":
            self.build()
        else:
            self.load_model()
    
    def load_data(self):
        """
            Not relevant for RL models, suggest a name change
        """
        pass


    def build(self):
        self.model = PPO(self.policy, 
                        self.env, 
                        device=self.device, 
                        tensorboard_log=self.log_path)

    def load_model(self):

        self.model = PPO.load(self.saved_model_path)

    def train(self):
        # will be moved to executor once callbacks are in place 
        self.model.learn(total_timesteps=self.num_timesteps)
        self.model.save(self.saved_model_path)

    def evaluate(self):
        # will be moved to evaluator later 
        curr_obs = self.env.reset()
        curr_step_num = 0 
        while True:
            action, _ = self.model.predict(curr_obs)
            # print(action)
            curr_obs, _, is_finished, info = self.env.step(action) 

            curr_step_num += 1
        
            if is_finished[0]:
                # curr_obs = env.reset()
                break
            elif curr_step_num > 100000:
                print("Long loop")
                # env.envs[0]._rep.final_map = np.copy(env.envs[0]._rep._map)
                break

        # write data file from final_map
        ut.write_curr_obs_to_dir_path(self.env.envs[0].rep.final_map, 
                        self.animations_path, 
                        curr_step_num,
                        self.lego_blocks_dims_dict
                        )