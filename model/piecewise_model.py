import os 
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import numpy as np


import gym
import torch
import torch.nn as nn

# local imports
from .base_model import BaseModel
from utils import utils as ut 
from gym_pcgrl.envs.pcgrl_env_3d_piecewise import LegoPCGEnv3DPiecewise

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        #self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        
        self.logger.record("train/reward_episode_end", self.model.env.envs[0].reward_history[-1])
        if len(self.model.env.envs[0].reward_history) > 2:
            self.logger.record("train/reward_step", (self.model.env.envs[0].reward_history[-1]-self.model.env.envs[0].reward_history[-2]))


        
            #self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        #self.logger.record("reward", self.model.ep_info_buffer[-1], exclude=("stdout", "log", "json", "csv"))
        #self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True



class LegoModelPiecewise(BaseModel):

    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)

        # unpack config
        self.train_config = self.config.train
        self.model_config = self.config.model
        self.policy = self.train_config["policy"]
        self.num_timesteps = self.train_config["num_episodes"]*self.train_config["reps_per_episode"]*self.train_config["num_of_blocks"]
        self.lego_blocks_dims_dict = self.train_config["LegoBlockDimensions"]
        self.cnn_output_channels = self.model_config["cnn_output_channels"]
        
        if not os.path.exists(self.model_config["log_path"]):
            os.mkdir(self.model_config["log_path"])
    
        savedir =  self.model_config["log_path"] 
        if self.train_config["controllable"]:
            savedir += "controllable_"
        
        savedir += self.train_config["reward_param"] +"_"+self.model_config["features_extractor"] + "_" + self.train_config["action_space"]
        
        if self.train_config["scheduled_episodes"]:
            savedir += "_sched_"
        
        else:
            savedir += "_" + str(self.train_config["reps_per_episode"]) + "_passes_"

        savedir +=  str(self.num_timesteps) + "_ts_" + str(self.train_config["num_of_blocks"]) + "_blocks_"  + str(self.train_config["observation_size"]) + "_obs_" + str(self.cnn_output_channels) + "_chans"
        
        if self.train_config["punish"]:
            savedir += "_punish"+ "_"+ str(self.train_config["punish"])

        iter = 0
        while os.path.exists(savedir +"_" + str(iter)):
            iter += 1

        self.animations_path = savedir +"_" + str(iter) + "/"
        os.mkdir(self.animations_path)

        self.log_path = f"{os.getcwd()}/tb_logs/" + self.animations_path.split("/")[-2]

        print(self.log_path)
        print(self.animations_path.split("/")[-2])
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        else:
            shutil.rmtree(self.log_path)

        self.saved_model_path = self.animations_path +"/model/"

        self.model = None
        self.device = ut.get_device()
        self.env = DummyVecEnv([lambda: LegoPCGEnv3DPiecewise(self.train_config, self.animations_path, model=self)])

        #check_env(self.env)

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
        

        if self.model_config["features_extractor"] == "default":
            self.model = PPO(self.policy, 
                            self.env, 
                            device=self.device, 
                            n_steps = min(self.num_timesteps, self.train_config['num_of_blocks'] * self.train_config['reps_per_episode']),
                            batch_size =  min(self.num_timesteps, self.train_config['num_of_blocks'] * self.train_config['reps_per_episode']),
                            tensorboard_log=self.log_path)
        
        elif self.model_config["features_extractor"] == "cnn":

            policy_kwargs = dict(
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs={
                    "cnn_output_channels": self.cnn_output_channels
                },
            )
            #model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)

            self.model = PPO(self.policy, 
                            self.env, 
                            device=self.device, 
                            n_steps = min(self.num_timesteps, self.train_config['num_of_blocks'] * self.train_config['reps_per_episode']),
                            batch_size =  min(self.num_timesteps, self.train_config['num_of_blocks'] * self.train_config['reps_per_episode']),
                            tensorboard_log=self.log_path,
                            policy_kwargs = policy_kwargs)

    def load_model(self):

        self.model = PPO.load(self.saved_model_path)

    def train(self):
        # will be moved to executor once callbacks are in place 
        custom_callback = TensorboardCallback()
        self.model.learn(self.num_timesteps, reset_num_timesteps=False, callback = custom_callback)#, callback = TensorboardCallback)
        self.model.save(self.saved_model_path)

        self.evaluate()

    def evaluate(self):
        # will be moved to evaluator later 
        #self.model = self.load_model()
        curr_obs = self.env.reset()
        curr_step_num = 0 

        ut.save_arrangement(
                self.env.envs[0].rep.blocks, 
                self.animations_path, 
                curr_step_num, 
                self.env.envs[0].reward_history[-1], 
                render = True)

        while True:

            action, _ = self.model.predict(curr_obs)
            curr_obs, _, is_finished, info = self.env.step(action) 

            curr_step_num += 1

            ut.save_arrangement(
                self.env.envs[0].rep.blocks, 
                self.animations_path, 
                curr_step_num, 
                None, 
                self.env.envs[0].reward_history,
                render = True)
            

            if is_finished[0]:
                # curr_obs = env.reset()
                break
            elif curr_step_num > 1000:
                print("Long loop")
                # env.envs[0]._rep.final_map = np.copy(env.envs[0]._rep._map)
                break

        
        ut.animate(self.animations_path)

        
        print(self.env.envs[0].reward_history)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param cnn_output_channels: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, cnn_output_channels):
        super(CustomCNN, self).__init__(observation_space, features_dim = cnn_output_channels)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, cnn_output_channels//2, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv3d(cnn_output_channels//2, cnn_output_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        #TO DO: fix linear layer thing -it doesn't fit unless the 128 is 128 instead of actual features dim. why??
        self.linear = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_channels):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim = 1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "map":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = CustomCNN(subspace, cnn_output_channels = cnn_output_channels)
                total_concat_size += 128
            elif key == "block_dims":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[1], 16))
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        
        return torch.cat(encoded_tensor_list, dim=1)



class ControllableCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_channels):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim = 1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "map":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = CustomCNN(subspace, cnn_output_channels = cnn_output_channels)
                total_concat_size += 128
            elif key == "block_dims":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[1], 16))
                total_concat_size += 16
            elif key == "goal_space":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[1], 16))
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        
        return torch.cat(encoded_tensor_list, dim=1)