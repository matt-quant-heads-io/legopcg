from gym.utils import seeding
from gym_pcgrl.envs.helper import gen_random_map
import numpy as np

TILES_MAP = {
    "g": "door",
    "+": "key",
    "A": "player",
    "1": "bat",
    "2": "spider",
    "3": "scorpion",
    "w": "solid",
    ".": "empty",
}

INT_MAP = {
    "empty": 0,
    "solid": 1,
    "player": 2,
    "key": 3,
    "door": 4,
    "bat": 5,
    "scorpion": 6,
    "spider": 7,
}

# For hashing maps to avoid duplicate goal states
CHAR_MAP = {
    "door": "a",
    "key": "b",
    "player": "c",
    "bat": "d",
    "spider": "e",
    "scorpion": "f",
    "solid": "g",
    "empty": "h",
}

# Reads in .txt playable map and converts it to string[][]
def to_2d_array_level(file_name):
    level = []

    with open(file_name, "r") as f:
        rows = f.readlines()
        for row in rows:
            new_row = []
            for char in row:
                if char != "\n":
                    new_row.append(TILES_MAP[char])
            level.append(new_row)

    # Remove the border
    truncated_level = level[1 : len(level) - 1]
    level = []
    for row in truncated_level:
        new_row = row[1 : len(row) - 1]
        level.append(new_row)
    return level


# Converts from string[][] to 2d int[][]
def int_arr_from_str_arr(map):
    int_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(INT_MAP[map[row_idx][col_idx]])
        int_map.append(new_row)
    return int_map


"""
The base class of all the representations
"""


class Representation:
    """
    The base constructor where all the representation variable are defined with default values
    """

    def __init__(self):
        self._random_start = True
        self._map = None
        self._old_map = None

        self.seed()

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int: the used seed (same as input if not None)
    """

    def seed(self, seed=None):
        self._random, seed = seeding.np_random(seed)
        return seed

    """
    Resets the current representation

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """

    def reset(self, width, height, prob):
        if self._random_start:
            # print(f"self._random is {self._random}")
            # print("random_start inside representation.py is set to True")
            # print(f"inside representation.reset()")
            self._map = gen_random_map(self._random, width, height, prob)
            # print(f"self._map is {self._map}")
            # print(f"self._map type is {type(self._map)}")
            # print(f"self._map[0] type is {type(self._map[0])}")
            # import numpy as np
            # self._map = np.array([[2, 6, 5, 4, 3, 1, 4, 0, 6, 1, 2], [0, 0, 1, 7, 0, 5, 4, 5, 0, 1, 0], [0, 6, 6, 3, 0, 0, 5, 1, 6, 3, 1], [7, 1, 0, 7, 6, 5, 7, 7, 6, 1, 4], [0, 1, 6, 0, 0, 1, 7, 5, 7, 4, 1], [0, 7, 6, 0, 0, 3, 0, 0, 3, 1, 6], [3, 7, 4, 0, 6, 2, 0, 1, 1, 6, 0]])

            self._old_map = self._map.copy()
            # import sys
            # sys.exit(0)
        else:
            # temp_map = int_arr_from_str_arr(to_2d_array_level(f'/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories_const_generated/narrow/init_maps_lvl1/init_map_0.txt'))
            start_map = "init_map_47.txt"
            temp_map = int_arr_from_str_arr(
                to_2d_array_level(
                    f"/gym-pcgil/gym_pcgil/exp_trajectories_const_generated/narrow_greedy/init_maps_lvl4/{start_map}"
                )
            )
            new_map = []
            for row in temp_map:
                new_map.append(np.array(row))

            self._map = np.array(new_map)
            # print(f"HIIIII representation.reset()")
            # import sys
            # sys.exit(0)
            self._old_map = self._map.copy()

    """
    Adjust current representation parameter

    Parameters:
        random_start (boolean): if the system will restart with a new map or the previous map
    """

    def adjust_param(self, **kwargs):
        self._random_start = kwargs.get("random_start", self._random_start)

    """
    Gets the action space used by the representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        ActionSpace: the action space used by that representation
    """

    def get_action_space(self, width, height, num_tiles):
        raise NotImplementedError("get_action_space is not implemented")

    """
    Get the observation space used by the representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        ObservationSpace: the observation space used by that representation
    """

    def get_observation_space(self, width, height, num_tiles):
        raise NotImplementedError("get_observation_space is not implemented")

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment
    """

    def get_observation(self):
        raise NotImplementedError("get_observation is not implemented")

    """
    Update the representation with the current action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """

    def update(self, action):
        raise NotImplementedError("update is not implemented")

    """
    Modify the level image with any special modification based on the representation

    Parameters:
        lvl_image (img): the current level_image without modifications
        tile_size (int): the size of tiles in pixels used in the lvl_image
        border_size ((int,int)): an offeset in tiles if the borders are not part of the level

    Returns:
        img: the modified level image
    """

    def render(self, lvl_image, tile_size, border_size):
        return lvl_image
