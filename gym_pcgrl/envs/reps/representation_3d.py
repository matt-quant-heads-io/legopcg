from gym.utils import seeding
from gym_pcgrl.envs.helper import gen_random_map_3d
import numpy as np 

class Representation3D:
    """
    The base class of all the representations
    """
    def __init__(self):
        """
        The base constructor where all the representation variable are defined with 
        default values
        """
        self._random_start = True
        self._map = None
        self._old_map = None
        self.render_map = None
        self.final_map = None
        self.lego_block_ids = None
        self.lego_block_dimensions_dict = None
        # self.brick_locations = {}
        self.num_bricks = None
        self.brick_details = []
        self.y = 0 
        self.x = 0 
        self.z = 0
        self.punish = False
        self.brick_added = False
        self.predicted_location = None

        self.seed()

    def seed(self, seed=None):
        """
        Seeding the used random variable to get the same result. If the seed is None,
        it will seed it with random start.

        Parameters:
            seed (int): the starting seed, if it is None a random seed number is used.

        Returns:
            int: the used seed (same as input if not None)
        """
        self._random, seed = seeding.np_random(seed)
        return seed

    def reset(self, **kwargs):
        """
        Resets the current representation

        Parameters:
            Kwargs from parent configs
        """
        # print(kwargs)
        # kwargs = kwargs["kwargs"]
        height, width, depth = kwargs.get("grid_dimensions")
        self.lego_block_ids = kwargs.get("lego_block_ids")
        self.lego_block_dimensions_dict = kwargs.get("lego_block_dims")
        self.num_bricks = kwargs.get("total_bricks")

        self._map = gen_random_map_3d(self._random, width, height, depth)
        self.render_map = gen_random_map_3d(self._random, width, height, depth)
        self.y = 0 
        self.x = 0 
        self.z = 0
        self.brick_details = []
        self.predicted_location = None
        # self.brick_locations = {}

    def adjust_param(self, **kwargs):
        """
        Adjust current representation parameter

        Parameters:
            random_start (boolean): if the system will restart with a new map or the previous map
        """
        self._random_start = kwargs.get('random_start', self._random_start)

    def get_action_space(self, width, height, num_tiles):
        """
        Gets the action space used by the representation

        Parameters:
            width: the current map width
            height: the current map height
            num_tiles: the total number of the tile values

        Returns:
            ActionSpace: the action space used by that representation
        """
        raise NotImplementedError('get_action_space is not implemented')


    def get_observation_space(self, width, height, num_tiles):
        """
        Get the observation space used by the representation

        Parameters:
            width: the current map width
            height: the current map height
            num_tiles: the total number of the tile values

        Returns:
            ObservationSpace: the observation space used by that representation
        """
        raise NotImplementedError('get_observation_space is not implemented')

    def get_observation(self):
        """
        Get the current representation observation object at the current moment

        Returns:
            observation: the current observation at the current moment
        """
        raise NotImplementedError('get_observation is not implemented')

    def update(self, action):
        """
        Update the representation with the current action

        Parameters:
            action: an action that is used to advance the environment (same as action space)

        Returns:
            boolean: True if the action change the map, False if nothing changed
        """
        raise NotImplementedError('update is not implemented')

    def render(self, lvl_image, tile_size, border_size):
        """
        Modify the level image with any special modification based on the representation

        Parameters:
            lvl_image (img): the current level_image without modifications
            tile_size (int): the size of tiles in pixels used in the lvl_image
            border_size ((int,int)): an offeset in tiles if the borders are not part of the level

        Returns:
            img: the modified level image
        """
        return lvl_image

    def _is_valid_location(self, tile_y, tile_x, tile_z):

        grid_width = self._map.shape[1]
        
        if self.x + tile_x >= grid_width:
            return False

        if self.y + tile_y >= grid_width:
            return False 

        if self.z + tile_z >= grid_width:
            return False
        
        # if the location underneath the current location is empty
        # i.e. brick will hand in the air so this is not a valid location

        # if self.y - 1 >= 0 and self._map[self.y-1][self.x][self.z] == 0:
        #     return False

        if self.y - 1 >= 0:
            # empty_underneath = True
            # for j in range(0, tile_x):
            #     for k in range(0, tile_z):
            #         if self._map[self.y - 1][self.x+j][self.z+k] != 0:
            #             empty_underneath = False
            #             break
            
            # if empty_underneath:
            #     return False

            if np.all(self._map[self.y-1, self.x:self.x+tile_x, self.z:self.z+tile_z] == 0):
                return False

        # if agent is trying to place a brick in non-empty location
        # remove the existing brick and make space for the new one
        # Note: This approach does not work for turtle representation as it gets
        # stuck in a loop of removing and placing bricks at the same location

        if (self._map[self.y][self.x][self.z] != 0):
            return False
            # brick_type = int(self._map[self.y][self.x][self.z])
            # # get the original location where brick was placed
            # o_y, o_x, o_z = self.brick_locations[(self.y,self.x,self.z)]

            # # print(brick_type)
            # tile_x, tile_y, tile_z = self.lego_block_dimensions_dict[self.lego_block_ids[brick_type]] 
            # for j in range(0, tile_x):
            #     for k in range(0, tile_z):
            #         if self._map[o_y][o_x+j][o_z+k] == brick_type:
            #             self._map[o_y][o_x+j][o_z+k] = 0 

        return True
