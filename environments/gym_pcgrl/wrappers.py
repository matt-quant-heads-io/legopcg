import gym
import gym_pcgrl

import numpy as np
import math
import os
import sys

# clean the input action
get_action = lambda a: a.item() if hasattr(a, "item") else a
# unwrap all the environments and get the PcgrlEnv
get_pcgrl_env = (
    lambda env: env if "PcgrlEnv" in str(type(env)) else get_pcgrl_env(env.env)
)

"""
Return a Box instead of dictionary by stacking different similar objects

Can be stacked as Last Layer
"""


class ToImage(gym.Wrapper):
    def __init__(self, game, names, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)
        self.shape = None
        depth = 0
        max_value = 0
        for n in names:
            assert (
                n in self.env.observation_space.spaces.keys()
            ), "This wrapper only works if your observation_space is spaces.Dict with the input names."
            if self.shape == None:
                self.shape = self.env.observation_space[n].shape
            new_shape = self.env.observation_space[n].shape
            depth += 1 if len(new_shape) <= 2 else new_shape[2]
            assert (
                self.shape[0] == new_shape[0] and self.shape[1] == new_shape[1]
            ), "This wrapper only works when all objects have same width and height"
            if self.env.observation_space[n].high.max() > max_value:
                max_value = self.env.observation_space[n].high.max()
        self.names = names

        self.observation_space = gym.spaces.Box(
            low=0, high=max_value, shape=(self.shape[0], self.shape[1], depth)
        )

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        final = np.empty([])
        for n in self.names:
            if len(final.shape) == 0:
                final = obs[n].reshape(self.shape[0], self.shape[1], -1)
            else:
                final = np.append(
                    final, obs[n].reshape(self.shape[0], self.shape[1], -1), axis=2
                )
        return final


"""
Transform any object in the dictionary to one hot encoding

can be stacked
"""


class OneHotEncoding(gym.Wrapper):
    def __init__(self, game, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert (
            name in self.env.observation_space.spaces.keys()
        ), "This wrapper only works for representations thave have a {} key".format(
            name
        )
        self.name = name

        self.observation_space = gym.spaces.Dict({})
        for (k, s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        new_shape = []
        shape = self.env.observation_space[self.name].shape
        self.dim = (
            self.observation_space[self.name].high.max()
            - self.observation_space[self.name].low.min()
            + 1
        )
        for v in shape:
            new_shape.append(v)
        new_shape.append(self.dim)
        self.observation_space.spaces[self.name] = gym.spaces.Box(
            low=0, high=1, shape=new_shape, dtype=np.uint8
        )

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        old = obs[self.name]
        # print(f"old is {old}")
        obs[self.name] = np.eye(self.dim)[old]
        # print(f"self.dim is {self.dim}")
        # print(f"transformed is {np.eye(self.dim)[old]}")
        return obs


"""
Transform the input space to a 3D map of values where the argmax value will be applied

can be stacked
"""


class ActionMap(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self._count = 0
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert (
            "map" in self.env.observation_space.spaces.keys()
        ), "This wrapper only works if you have a map key"
        self.old_obs = None
        self.one_hot = len(self.env.observation_space["map"].shape) > 2
        # print(f"self.one_hot is {self.one_hot}")
        # print(f"self.env._rep._map is {self.env._rep._map}")
        # print(f"self.env.observation_space.spaces is {self.env.observation_space.spaces}")
        w, h, dim = 0, 0, 0
        if self.one_hot:
            h, w, dim = self.env.observation_space["map"].shape
        else:
            h, w = self.env.observation_space["map"].shape
            dim = self.env.observation_space["map"].high.max()
        self.h = self.unwrapped.h = h
        self.w = self.unwrapped.w = w
        self.dim = self.unwrapped.dim = self.env.get_num_tiles()
        # self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(h,w,dim))
        self.action_space = gym.spaces.Discrete(h * w * self.dim)

    def reset(self):
        self.old_obs = self.env.reset()
        return self.old_obs

    def step(self, action):
        # y, x, v = np.unravel_index(np.argmax(action), action.shape)
        y, x, v = np.unravel_index(action, (self.h, self.w, self.dim))
        if "pos" in self.old_obs:
            o_x, o_y = self.old_obs["pos"]
            if o_x == x and o_y == y:
                obs, reward, done, info = self.env.step(v)
            else:
                o_v = self.old_obs["map"][o_y][o_x]
                if self.one_hot:
                    o_v = o_v.argmax()
                obs, reward, done, info = self.env.step(o_v)
        else:
            obs, reward, done, info = self.env.step([x, y, v])
        self.old_obs = obs
        self._count += 1
        # print(f"obs in ActionMap.step() (called {self._count} times) is {obs}")
        return obs, reward, done, info


"""
Crops and centers the view around the agent and replace the map with cropped version
The crop size can be larger than the actual view, it just pads the outside
This wrapper only works on games with a position coordinate

can be stacked
"""


class Cropped(gym.Wrapper):
    def __init__(self, game, crop_size, pad_value, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert (
            "pos" in self.env.observation_space.spaces.keys()
        ), "This wrapper only works for representations thave have a position"
        assert (
            name in self.env.observation_space.spaces.keys()
        ), "This wrapper only works if you have a {} key".format(name)
        assert (
            len(self.env.observation_space.spaces[name].shape) == 2
        ), "This wrapper only works on 2D arrays."
        self.name = name
        self.size = crop_size
        self.pad = crop_size // 2
        self.pad_value = pad_value
        # print(f"pad_value is {pad_value}")

        self.observation_space = gym.spaces.Dict({})
        for (k, s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        high_value = self.observation_space[self.name].high.max()
        self.observation_space.spaces[self.name] = gym.spaces.Box(
            low=0, high=high_value, shape=(crop_size, crop_size), dtype=np.uint8
        )

    def step(self, action):
        action = get_action(action)
        # print(f"self.env.action_space is {self.env.action_space}")
        # print(f"{self.env.action_space.high}")
        # import sys
        # sys.exit(0)
        # if action < 0 or action > 7:
        # print(f"action is {action}")
        # print(f"action is: {action}")
        obs, reward, done, info = self.env.step(action)
        # print(f"obs is: {obs}")
        obs = self.transform(obs)
        # print(f"obs after transform: {obs}")
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        map = obs[self.name]
        # print(f"map in transform is {map}")
        x, y = obs["pos"]
        # print(f"self.pad_value is {self.pad_value}")
        # print(f"self.pad is {self.pad}")
        # print(f"x,y is {x},{y}")
        # print(f"self.size is {self.size}")
        # View Centering
        padded = np.pad(map, self.pad, constant_values=self.pad_value)
        cropped = padded[y : y + self.size, x : x + self.size]
        obs[self.name] = cropped
        # print(f"cropped is {cropped}")

        return obs


################################################################################
#   Final used wrappers for the experiments
################################################################################

"""
The wrappers we use for narrow and turtle experiments
"""


class CroppedImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, crop_size, **kwargs):
        # print(f"game is {game}")
        # print(f"kwargs is {kwargs}")
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # TODO:
        # Cropping the map to the correct crop_size
        # Vary the crop size to get different sizes: 3, 5, 10, 20 --> measure effect of crop size on generalization
        # (test random)
        # Test with varying training_trajectories lengths
        env = Cropped(
            self.pcgrl_env, crop_size, self.pcgrl_env.get_border_tile(), "map"
        )
        # print(f"crop_size is {crop_size}")
        # Transform to one hot encoding if not binary
        if "binary" not in game:
            env = OneHotEncoding(env, "map")
            # print(f"env is {env}")
        # Indices for flatting
        flat_indices = ["map"]
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)


"""
Similar to the previous wrapper but the input now is the index in a 3D map (height, width, num_tiles) of the highest value
Used for wide experiments
"""


class ActionMapImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Indices for flatting
        flat_indices = ["map"]
        env = self.pcgrl_env
        # Add the action map wrapper
        env = ActionMap(env)
        # Transform to one hot encoding if not binary
        if "binary" not in game:
            env = OneHotEncoding(env, "map")
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)
