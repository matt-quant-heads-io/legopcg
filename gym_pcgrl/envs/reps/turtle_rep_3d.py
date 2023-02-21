import numpy as np
from gym import spaces

# local imports
from gym_pcgrl.envs.reps.representation_3d import Representation3D


class TurtleRepresntation3D(Representation3D):
    def __init__(self):
        # To keep count of number of bricks currently in the grid space
        # self.num_of_bricks = 100
        # self.num_of_bricks = total_bricks
        super().__init__()
        self.punish: bool = False
        self.final_map = None
        self.old_location = [] 
        self.y = 0
        self.x = 5
        self.z = 5
        self.max_height = None # max y starting from 0
        self.max_width = None  # max x starting from 0
        self.max_depth = None  # max z starting from 0
        self.num_of_bricks = 0


    # def generate_random_map(self):
    #     # TODO: This needs to be random rather
    #     map = np.zeros(shape=(self.grid_width,self.grid_height,self.grid_depth))
    #     return map

    # def empty_map(self):
    #     map = np.zeros(shape=(self.grid_width,self.grid_height,self.grid_depth))
    #     return map
        
    def reset(self, height, width, depth, total_bricks):
        # Reset all internal data structures being used
        # self.final_map = np.copy(self._map)
        super().reset(height, width, depth)
        # self._map = self.empty_map()
        # self.num_of_bricks = self.total_bricks
        # self.final_map = None
        self.punish = False
        self.y = 0
        self.x = 5
        self.z = 5
        self.max_height = height - 1
        self.max_width = width - 1
        self.max_depth = depth - 1
        self.num_of_bricks = total_bricks

        # return self._map

    def get_action_space(self, num_actions=6):
        """
            default: left, right, up, down, rise and fall
        """
        return spaces.Discrete(num_actions)
    
    def get_observation_space(self, width=10, height=10, depth=10):
        """
            default: 3-D grid of 10x10x10
        """
        return spaces.Discrete(width * depth * height)
    
    def get_observation(self) -> int:
        """Return state number"""
        return self.y * 100 + self.x * 10 + self.z
        
    def update(self, action):
        """
            Take action, determine whether to reward or punish the action. 
        """
        self.punish = False
        self.old_location = [self.y,self.x,self.z]
        self._move_to_state(action)

        if self._is_location_valid():

            if self._map[self.y][self.x][self.z] != 1  and self._is_connected():
                # fill the number in first place 
                # for now, fill with 1x1 box
                self._map[self.y][self.x][self.z] = 1
                # fill the rest with -1 (special value)
                # for _ in range(1, box_type):
                #     self._map[z][y][x] = -1 
                #     x +=1 
            
                self.num_of_bricks -= 1
            else:
                self.punish = True 
        else:
            self.punish = True
        # return 1 
    
    def _move_to_state(self, action):
        """
            Check whether grid has enough space to accommodate 
            the predicted block or not.
        """
        # left = 0 , right= 1, up = 2, down = 3, rise = 4, fall = 5
        if action == 0:
            self.x = max(self.x - 1, 0)
        elif action == 1:
            # self.x = min(self.x + 1, 9)
            self.x = min(self.x + 1, self.max_width)
        elif action == 2:
            self.z = max(self.z - 1, 0)
        elif action == 3:
            # self.z = min(self.z + 1, 9)
            self.z = min(self.z + 1, self.max_depth)
        elif action == 4:
            # self.y = min(self.y + 1, 9)
            self.y = min(self.y + 1, self.max_height)
        elif action == 5:
            self.y = max(self.y - 1, 0)
        else:
            print("Invalid Action!!")


    def _is_location_valid(self):
        """
            Current state should not be equal to old state 
        """
        return [self.y, self.x, self.z] != self.old_location
    
    def _is_connected(self):
        """
            Check whether the new block is connected 
            to the old block along any of the axis. 
            Return True if it is otherwise return False.
        """

        if self.old_location:
            # blocks are adjacent if difference is 1 across any of the axis
            a = np.array([self.y, self.x, self.z]) - np.array(self.old_location)

            if any(a == 1) or any(a == -1):
                return True
            else:
                return False
        else:
            return True

    # def get_state_number(self):

    #     return self.y * 100 + self.x * 10 + self.z

