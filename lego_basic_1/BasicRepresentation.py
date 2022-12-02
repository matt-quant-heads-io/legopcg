from gym import spaces
import numpy as np
from collections import OrderedDict
from gym.utils import seeding

class BasicRepresentation:
    def __init__(self):

        # Starting point in the 3-D space for Lego Generation is random
        self._random_start = True

        # _map data structure is the Observation Space 
        # 3D array of Actions for each coordinate
        # Actions here are Lego Block Types
        self._map = None

        # Previous Observation
        self._old_map = None
        
        # Seed to initialize the random-number generator        
        self.seed()

        # ?
        # self._random_tile = True

    # If the seed is None, it will seed it with random start
    def seed(self, seed=None):
        self._random, seed = seeding.np_random(seed)
        return seed

    """
    Resets the current representation where it resets the parent and the current
    modified location

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, width, height, depth, prob):
        if self._random_start or self._old_map is None:

            # Generate Random Observation space
            self._map = gen_random_map(self._random, width, height, depth, prob)
            
            # Reset old map
            self._old_map = self._map.copy()
        else:
            #
            self._map = self._old_map.copy()

        # Reset x,y,z coordinates
        self._x = self._random.randint(width)
        self._y = self._random.randint(height)
        self._z = self._random.randint(depth)
    
    def gen_random_map(random, width, height, depth, prob):
        map = random.choice(list(prob.keys()),size=(height,width,depth),p=list(prob.values())).astype(np.uint8)
        return map

    """
    Gets the action space used by the narrow representation
    """
    def get_action_space(self, num_tiles):
        return spaces.Discrete(num_tiles + 1)

    """
    Get the observation space used by the narrow representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Dict: the observation space used by that representation. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observation_space(self, width, height, depth, num_tiles):
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1, depth-1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width,depth))
        })

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._x, self._y, self._z], dtype=np.uint8),
            "map": self._map.copy()
        })

    """
    Adjust the current used parameters

    Parameters:
        random_start (boolean): if the system will restart with a new map (true) or the previous map (false)
        random_tile (boolean): if the system will move between tiles random (true) or sequentially (false)
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._random_tile = kwargs.get('random_tile', self._random_tile)

    """
    Update the Basic representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        change = 0
        if action > 0:
            change += [0,1][self._map[self._y][self._x] != action-1]
            self._map[self._y][self._x] = action-1
        if self._random_tile:
            self._x = self._random.randint(self._map.shape[1])
            self._y = self._random.randint(self._map.shape[0])
        else:
            self._x += 1
            if self._x >= self._map.shape[1]:
                self._x = 0
                self._y += 1
                if self._y >= self._map.shape[0]:
                    self._y = 0
        return change, self._x, self._y

    """
    Modify the level image with a red rectangle around the tile that is
    going to be modified

    Parameters:
        lvl_image (img): the current level_image without modifications
        tile_size (int): the size of tiles in pixels used in the lvl_image
        border_size ((int,int)): an offeset in tiles if the borders are not part of the level

    Returns:
        img: the modified level image
    """
    def render(self, lvl_image, tile_size, border_size):
        x_graphics = Image.new("RGBA", (tile_size,tile_size), (0,0,0,0))
        for x in range(tile_size):
            x_graphics.putpixel((0,x),(255,0,0,255))
            x_graphics.putpixel((1,x),(255,0,0,255))
            x_graphics.putpixel((tile_size-2,x),(255,0,0,255))
            x_graphics.putpixel((tile_size-1,x),(255,0,0,255))
        for y in range(tile_size):
            x_graphics.putpixel((y,0),(255,0,0,255))
            x_graphics.putpixel((y,1),(255,0,0,255))
            x_graphics.putpixel((y,tile_size-2),(255,0,0,255))
            x_graphics.putpixel((y,tile_size-1),(255,0,0,255))
        lvl_image.paste(x_graphics, ((self._x+border_size[0])*tile_size, (self._y+border_size[1])*tile_size,
                                        (self._x+border_size[0]+1)*tile_size,(self._y+border_size[1]+1)*tile_size), x_graphics)
        return lvl_image
