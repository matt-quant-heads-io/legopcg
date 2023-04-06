# import os 
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import DummyVecEnv
import numpy as np 

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
            # env = LegoPCGEnv3D(self.train_config, representation="wide3d")
            # env = LegoPCGEnv3D(self.train_config, representation="turtle3d")
            env = LegoPCGEnv3D(self.train_config, representation="narrow3d")
            env = Cropped3D(env)
            return env

        return thunk

    def evaluate(self):
        # will be moved to evaluator later 
        # self.model = self.load_model()
        ut.cleanup_dir(self.animations_path)
        curr_obs = self.env.reset()
        # lego_block_coords = []

        curr_step_num = 0 
        envs_not_processed = True

        while envs_not_processed:
            action, _ = self.model.predict(curr_obs)
            curr_obs, _, is_finished, info = self.env.step(action) 

            curr_step_num += 1

            for env_num, info_dict in enumerate(info):
                # generate files for leocad rendering
                if info_dict["brick_added"]: 
                    ut.render_in_leocad(self.animations_path,
                                        env_num, 
                                        info_dict["block_details"])

                if is_finished[env_num]:
                    # write for the environment which is finished
                    self._write_ldr(env_num, 
                                    self.env.envs[env_num].rep.final_map, 
                                    )

                elif curr_step_num > 70000:
                    print("Long loop")
                    print(info)
                    # write only for the first environment
                    self._write_ldr(0,
                                    self.env.envs[0].rep.render_map,
                                    )

                    return
            if is_finished[0]:
                envs_not_processed = False

        ut.create_gif(self.animations_path)            
    
    def _write_ldr(self, i, env_map):
        if isinstance(env_map, np.ndarray):
            ut.write_curr_obs_to_dir_path(env_map, 
                            self.animations_path, 
                            f"env_{i}",
                            self.lego_blocks_dims_dict)
