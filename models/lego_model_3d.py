# import os
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import DummyVecEnv
from typing import Callable, Dict, Tuple
import numpy as np

# import os
import sys
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

from gym import spaces

# local imports
from .ppo_model import PPOModel
from utils import utils as ut
from gym_envs.gym_pcgrl.envs.pcgrl_env_3d import LegoPCGEnv3D
from gym_envs.gym_pcgrl.wrappers import Cropped3D


class LegoNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.hidden_dim1 = feature_dim * 2
        self.hidden_dim2 = last_layer_dim_pi * 2

        print("feature_dim: ", feature_dim)
        print("hidden_dim1: ", self.hidden_dim1)
        print("hidden_dim2: ", self.hidden_dim2)
        print("last_layer_dim_pi: ", last_layer_dim_pi)
        print("last_layer_dim_vf: ", last_layer_dim_vf)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.ReLU(),
            # nn.Linear(self.hidden_dim1, self.hidden_dim1),
            # nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, last_layer_dim_pi),
            nn.ReLU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.ReLU(),
            # nn.Linear(self.hidden_dim1, self.hidden_dim1),
            # nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class LegoActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        print("Using custom LegoActorCriticPolicy")

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LegoNetwork(self.features_dim)


class LegoModel3D(PPOModel):
    def __init__(self, cfg: Dict, mode="train"):

        super().__init__(cfg)

        self.lego_blocks_dims_dict = self.train_config["lego_block_dims"]

        self.env = self.get_vector_env(self._make_env)

        # override policy value
        self.policy = LegoActorCriticPolicy
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
            env = LegoPCGEnv3D(self.train_config, representation="wide3d")
            # env = LegoPCGEnv3D(self.train_config, representation="turtle3d")
            # env = LegoPCGEnv3D(self.train_config, representation="narrow3d")
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
        envs_not_processed = [True for _ in range(self.env.num_envs)]

        while any(envs_not_processed):
            action, _ = self.model.predict(curr_obs)
            curr_obs, _, is_finished, info = self.env.step(action)

            curr_step_num += 1

            for env_num, info_dict in enumerate(info):
                # generate files for leocad rendering
                if info_dict["brick_added"] and envs_not_processed[env_num]:
                    ut.render_in_leocad(
                        self.animations_path, env_num, info_dict["brick_details"]
                    )

                # if is_finished[env_num]:
                if info_dict["num_of_bricks"] <= 0 and envs_not_processed[env_num]:
                    # print("Writing ldr")
                    envs_not_processed[env_num] = False
                    # write for the environment which is finished
                    # self._write_ldr(env_num,
                    #                 self.env.envs[env_num].rep.final_map,
                    #                 )

                elif curr_step_num > 70000:
                    print("Long loop")
                    print(info)
                    # write only for the first environment
                    # self._write_ldr(0,
                    #                 self.env.envs[0].rep.render_map,
                    #                 )

                    return

        # ut.create_gif(self.animations_path)

    def _write_ldr(self, i: int, env_map):
        if isinstance(env_map, np.ndarray):
            ut.write_curr_obs_to_dir_path(
                env_map, self.animations_path, f"env_{i}", self.lego_blocks_dims_dict
            )
