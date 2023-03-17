import os 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

# local imports
from .ppo_model import PPOModel
from utils import utils as ut 
from gym_pcgrl.envs.pcgrl_env_3d import LegoPCGEnv3D
from gym_pcgrl.wrappers import Cropped3D

class LegoModel3D(PPOModel):

    def __init__(self, cfg, mode="train"):

        super().__init__(cfg)

        self.lego_blocks_dims_dict = self.train_config["lego_block_dims"]

        self.env = self.get_vector_env(self._make_env)

        # train or load a trained model
        if mode == "train":
            self.build()
        else:
            self.load_model()
    
    def _make_env(self):
        def thunk():
            # env = gym.make(env_id)
            # env = gym.wrappers.RecordEpisodeStatistics(env)
            # if capture_video:
            #     if idx == 0:
            #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            # env.seed(seed)
            # env.action_space.seed(seed)
            # env.observation_space.seed(seed)
            env = LegoPCGEnv3D(self.train_config)
            env = Cropped3D(env)
            return env

        return thunk

    def evaluate(self):
        # will be moved to evaluator later 
        # self.model = self.load_model()
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
                return

        # write data file from final_map
        ut.write_curr_obs_to_dir_path(self.env.envs[0].rep.final_map, 
                        self.animations_path, 
                        curr_step_num,
                        self.lego_blocks_dims_dict
                        )