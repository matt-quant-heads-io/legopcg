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

    def get_action_space(self, **kwargs):
        height, width, depth = kwargs.get("grid_dimensions")
        total_bricks = len(kwargs.get("lego_block_ids"))
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

    def get_observation_space(self, **kwargs):
        height, width, depth = kwargs.get("crop_dimensions")
        num_tiles = len(kwargs.get("lego_block_ids"))
        return spaces.Box(low=0, high=num_tiles, shape=(height,width,depth))

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. A 2D array of tile numbers
    """

    def get_observation(self):

        return self._map


    def update(self, action):

        """
        Update the wide representation with the input action

        Parameters:
            action: an action that is used to advance the environment (same as action space)

        Returns:
            Bool: True if the action changes the map, False if nothing changed

        """

        self.punish = False
        self.brick_added = False
        # unpack action 
        self.y, self.x, self.z, brick_type = action   
        tile_x, _, _ = self.lego_block_dimensions_dict[self.lego_block_ids[brick_type]]

        # valid location if = 0 or
        # the tile can be placed within the bounds of the grid
        # for now the tiles are being placed along x-axis
        # perhaps in future their axis orientation can also 
        # be explored

        
        grid_width = self._map.shape[1]

        # if (self._map[y][x][z] == 0 and 
        #     x + tile_x <= grid_width):

        if (self._is_valid_location() and 
            self.x + tile_x <= grid_width):

            # standard code to be added in all representations
            if brick_type > 0:
                self.num_bricks -= 1
                self.brick_added = True
                self.block_details.append((self.y,self.x,self.z, brick_type)) 

            # fill the number in first place 
            self._map[self.y][self.x][self.z] = brick_type
            self.render_map[self.y][self.x][self.z] = brick_type

            # fill the rest with -1 (special value)
            for i in range(1, tile_x):
                self._map[self.y][self.x + i ][self.z] = brick_type
        else:
            self.punish = True

        return
    
    def _is_valid_location(self):
        
        if self._map[self.y][self.x][self.z] != 0:
            return False
        
        if self.y - 1 >= 0 and self._map[self.y-1][self.x][self.z] == 0:
            return False
        
        # if x - 1 >= 0 and self._map[y][x-1][z] == 0:
        #     return False

        # if x + 1 >= 10 and self._map[y][x+1][z] == 0:
        #     return False
        
        # if z - 1 >= 0 and self._map[y][x][z-1] == 0:
        #     return False

        # if z + 1 >= 10 and self._map[y][x][z+1] == 0:
        #     return False
                
        return True

        
