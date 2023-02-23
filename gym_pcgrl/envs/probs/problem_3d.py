from gym.utils import seeding
# from PIL import Image

class Problem3D:
    """
    The base class for all the 3D problems that can be handled by the interface
    """
    def __init__(self):
        """
        Constructor for the problem that initialize all the basic parameters
        """
        self.width = 10
        self.height = 10
        self.depth = 10 
        self.total_bricks = 100
        tiles = self.get_tile_types()
        # self._prob = []
        # for _ in range(len(tiles)):
        #     self._prob.append(1.0/len(tiles))

        # self._border_size = (1,1)
        # self._border_tile = tiles[0]
        # self._tile_size=16
        # self._graphics = None

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

    def reset(self, start_stats={}):
        """
        Resets the problem to the initial state and save the start_stats from the 
        starting map. Also, it can be used to change values between different 
        environment resets

        Parameters:
            start_stats (dict(string,any)): the first stats of the map
        """
        self._start_stats = start_stats

    def get_tile_types(self):
        """
        Get a list of all the different tile names

        Returns:
            string[]: that contains all the tile names
        """
        raise NotImplementedError('get_tile_types is not implemented')


    def adjust_param(self, **kwargs):
        """
        Adjust the parameters for the current problem

        Parameters:
            width (int): change the width of the problem level
            height (int): change the height of the problem level
            depth (int): change the depth of the problem level
        """
        self.width = kwargs.get('width', self.width)
        self.height = kwargs.get('height', self.height)
        self.depth = kwargs.get('depth', self.depth)
        
        # prob = kwargs.get('probs')
        # if prob is not None:
        #     for t in prob:
        #         if t in self._prob:
        #             self._prob[t] = prob[t]

    def get_stats(self, map):
        """
        Get the current stats of the map

        Returns:
            dict(string,any): stats of the current map to be used in the reward, 
            episode_over, debug_info calculations
        """
        raise NotImplementedError('get_stats is not implemented')

    def get_reward(self, new_stats, old_stats):
        """
        Get the current game reward between two stats

        Parameters:
            new_stats (dict(string,any)): the new stats after taking an action
            old_stats (dict(string,any)): the old stats before taking an action

        Returns:
            float: the current reward due to the change between the old map 
            stats and the new map stats
        """
        raise NotImplementedError('get_reward is not implemented')

    def get_episode_over(self, new_stats, old_stats):
        """
        Uses the stats to check if the problem ended (episode_over) which means reached
        a satisfying quality based on the stats

        Parameters:
            new_stats (dict(string,any)): the new stats after taking an action
            old_stats (dict(string,any)): the old stats before taking an action

        Returns:
            boolean: True if the level reached satisfying quality based on the stats and 
            False otherwise
        """
        raise NotImplementedError('get_episode_over is not implemented')

    def get_debug_info(self, new_stats, old_stats):
        """
        Get any debug information need to be printed

        Parameters:
            new_stats (dict(string,any)): the new stats after taking an action
            old_stats (dict(string,any)): the old stats before taking an action

        Returns:
            dict(any,any): is a debug information that can be used to debug what is
            happening in the problem
        """
        raise NotImplementedError('get_debug_info is not implemented')

    def render(self, map):
        """
        Get an image on how the map will look like for a specific map

        Parameters:
            map (string[][]): the current game map

        Returns:
            Image: a pillow image on how the map will look like using the problem
            graphics or default grey scale colors
        """
        # tiles = self.get_tile_types()
        # if self._graphics == None:
        #     self._graphics = {}
        #     for i in range(len(tiles)):
        #         color = (i*255/len(tiles),i*255/len(tiles),i*255/len(tiles),255)
        #         self._graphics[tile[i]] = Image.new("RGBA",(self._tile_size,self._tile_size),color)

        # fullwidth = len(map[0])+2*self._border_size[0]
        # fullheight = len(map)+2*self._border_size[1]
        # lvl_image = Image.new("RGBA", (fullwidth*self._tile_size, fullheight*self._tile_size), (0,0,0,255))
        # for y in range(fullheight):
        #     for x in range(self._border_size[0]):
        #         lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
        #         lvl_image.paste(self._graphics[self._border_tile], ((fullwidth-x-1)*self._tile_size, y*self._tile_size, (fullwidth-x)*self._tile_size, (y+1)*self._tile_size))
        # for x in range(fullwidth):
        #     for y in range(self._border_size[1]):
        #         lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
        #         lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, (fullheight-y-1)*self._tile_size, (x+1)*self._tile_size, (fullheight-y)*self._tile_size))
        # for y in range(len(map)):
        #     for x in range(len(map[y])):
        #         lvl_image.paste(self._graphics[map[y][x]], ((x+self._border_size[0])*self._tile_size, (y+self._border_size[1])*self._tile_size, (x+self._border_size[0]+1)*self._tile_size, (y+self._border_size[1]+1)*self._tile_size))
        # return lvl_image
        pass
