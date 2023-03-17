from gym_pcgrl.envs.reps.representation_3d import Representation3D
from gym import spaces
import numpy as np

"""
The wide representation where the agent can pick the tile position and tile value at each update.
"""


class WideRepresentation3D(Representation3D):
    """
    Initialize all the parameters used by that representation
    """

    def __init__(self):
        super().__init__()

    """
    Gets the action space used by the wide representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        MultiDiscrete: the action space used by that wide representation which
        consists of the x position, y position, and the tile value
    """

    def get_action_space(self, height, width, depth, total_bricks):
        return spaces.MultiDiscrete([height, width, depth, total_bricks])

    """
    Get the observation space used by the wide representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Box: the observation space used by that representation. A 2D array of tile numbers
    """

    def get_observation_space(self, height = 5, width=5, depth=5, num_tiles=4):
        # return spaces.Dict(
        #     {
        #         "map": spaces.Box(
        #             low=0, high=num_tiles - 1, dtype=np.uint8, shape=(height, width)
        #         )
        #     }
        # )
        # spaces.Box(low=0, high=4, shape=(5,5,5))
        return spaces.Box(low=0, high=num_tiles, shape=(height,width,depth))

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. A 2D array of tile numbers
    """

    def get_observation(self):
        # return {"map": self._map.copy()}
        return self._map


    def update(self, action):

        """
        Update the wide representation with the input action

        Parameters:
            action: an action that is used to advance the environment (same as action space)

        Returns:
            Bool: True if the action changes the map, False if nothing changed

        """

        punish = False
        # unpack action 
        y, x, z, tile_type = action   
        tile_x, _, _ = self.lego_block_dimensions_dict[self.lego_block_ids[tile_type]]

        # valid location if = 0 or
        # the tile can be placed within the bounds of the grid
        # for now the tiles are being placed along x-axis
        # perhaps in future their axis orientation can also 
        # be explored

        
        grid_width = self._map.shape[1]

        if (self._map[y][x][z] == 0 and 
            x + tile_x <= grid_width):
            
            # fill the number in first place 
            self._map[y][x][z] = tile_type
            self.render_map[y][x][z] = tile_type

            # fill the rest with -1 (special value)
            for i in range(1, tile_x):
                self._map[y][x + i ][z] = tile_type

            if tile_type > 0:
                self.num_bricks -= 1
        else:
            punish = True

        return punish