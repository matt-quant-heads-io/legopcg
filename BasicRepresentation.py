from gym import spaces
import numpy as np
from collections import OrderedDict
import random 
import LegoDimensions
import map_encodings

width = 10
depth = 10
height = 10

low = np.zeros(shape=(10,10,10))
high = np.ones(shape=(10,10,10))

class BasicRepresentation:
    def __init__(self):

        self._random_start = True
        self._random_tile = True
        self._map = self.gen_random_map()

        self._studs = self.empty_map()
        self._isStudConnected = self.empty_map()
        self._isRenderError = False

        self._x = np.random.randint(low=0, high=9)
        self._y = np.random.randint(low=0, high=9)
        self._z = np.random.randint(low=0, high=9)  
    
    def gen_random_map(self):
        # map = spaces.Box(low=low, high=high, shape=(10,10,10))
        map = np.zeros(shape=(10,10,10))
        # map[0][0][0] = 1
        return map

    def empty_map(self):
        map = np.zeros(shape=(10,10,10))
        return map
        
    def reset(self):
        self._map = self.gen_random_map()
        self._studs = self.empty_map()
        self._isStudConnected = self.empty_map()
        self._isRenderError = False

        # Reset x,y,z coordinates
        self._x = 0
        self._y = 0
        self._z = 0  

        return self._map

    def get_observation_space(self, width, height, depth):
        return self.get_observation()

    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._x, self._y, self._z], dtype=np.uint8),
            "map": self._map 
        })
        
    def update(self, action):
        change = 0
        self._isRenderError = False
        
        if action > 0:
            change += 1

            legoBlockName = map_encodings.onehot_index_to_str_map[action]
            xyzDims = LegoDimensions.LegoDimsDict[legoBlockName]
            lengthOfBlock = xyzDims[0]
            if self._x + lengthOfBlock <= self._map.shape[2]: # Check for < less than
                self._map[self._z][self._y][self._x] = action
                
                for l in range(lengthOfBlock):
                    self._studs[self._z][self._y][self._x + l] = 1 # Stud exists at this location -> 0,1,2 for a 1x3 block
                    
                    # Is there a stud right below the current location; if yes, mark it as connected
                    if ((self._y - 1) >= 0 and self._studs[self._z][self._y-1][self._x + l] == 1):
                        self._isStudConnected[self._z][self._y-1][self._x + l] = 1

            else:
                self._isRenderError = True

        # New Agent location 
        legoBlockName = map_encodings.onehot_index_to_str_map[action]
        if (legoBlockName != 'empty'):
            xyzDims = LegoDimensions.LegoDimsDict[legoBlockName]
            self._x += xyzDims[0] 
        else:
            self._x += 1

        if self._x >= self._map.shape[2]:
            self._x = 0
            self._z += 1
            
            if self._z >= self._map.shape[0]:
                self._z = 0
                self._y += 1

                if self._y >= self._map.shape[1]:
                    self._y = 0
                
        return change, self._x, self._y, self._z
