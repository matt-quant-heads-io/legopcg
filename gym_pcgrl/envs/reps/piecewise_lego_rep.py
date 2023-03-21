
import random
import numpy as np
from gym import spaces
from utils.utils import LegoDimsDict
import utils.utils as ut

# local imports
from gym_pcgrl.envs.reps.representation_3d import Representation3D 


class LegoBlock():
    def __init__(self, dimsNum, rep):
        super().__init__()
        self.dims = LegoDimsDict[dimsNum]
        self.x = 0
        self.y = random.randrange(rep.max_y)
        self.z = random.randrange(rep.max_z)
        self.rep = rep

        position = str(self.x) + str(self.y) + str(self.z)
        while position in rep.block_positions:
            self.y = random.randrange(rep.max_y)
            self.z = random.randrange(rep.max_z)

            position = str(self.x) + str(self.y) + str(self.z)

        rep.block_positions.append(position)

        self.next_x = self.x
        self.next_y = self.y
        self.next_z = self.z

        

    def reset(self):

        self.x = 0
        self.y = random.randrange(self.rep.max_y)
        self.z = random.randrange(self.rep.max_z)

        position = str(self.x) + str(self.y) + str(self.z)
        while position in self.rep.block_positions:
            self.y = random.randrange(self.rep.max_y)
            self.z = random.randrange(self.rep.max_z)

            position = str(self.x) + str(self.y) + str(self.z)

        self.rep.block_positions.append(position)


    



        


class PiecewiseRepresentation(Representation3D):
    def __init__(self, train_cfg):
        # To keep count of number of bricks currently in the grid space
        # self.num_of_bricks = 100
        # self.num_of_bricks = total_bricks
        super().__init__()
        
        self.punish: bool = False
        self.final_full_map = None
        self.height = 0
        self.old_height = 0
        self.observation_size = train_cfg["observation_size"]

        self.max_x= train_cfg["GridDimensions"][0]# max y starting from 0
        self.max_y = train_cfg["GridDimensions"][1]  # max x starting from 0
        self.max_z = train_cfg["GridDimensions"][2]  # max z starting from 0
        
        self.curr_block = 0
        self.num_of_blocks = train_cfg["num_of_blocks"]
        assert self.num_of_blocks <= self.max_x  * self.max_z

        self.blocks = []
        self.block_positions = []

        for i in range(self.num_of_blocks):
            self.curr_block = i
            block = LegoBlock("3005", self)
            self.blocks.append(block)
        self.curr_block = 0


        self._map = self.get_map()
        self._last_map = None
        self.final_full_map= self.get_full_map()
        self.step = 0

        #ut.save_map_piecewise(self.final_map, "/home/maria/dev/legopcg/animations", self.step, LegoDimsDict)



    # def generate_random_map(self):
    #     # TODO: This needs to be random rather
    #     map = np.zeros(shape=(self.grid_width,self.grid_height,self.grid_depth))
    #     return map

    # def empty_map(self):
    #     map = np.zeros(shape=(self.grid_width,self.grid_height,self.grid_depth))
    #     return map
        
    def reset(self):
        # Reset all internal data structures being used
        # self.final_map = np.copy(self._map)
        #super().reset(height, width, depth)
        self._last_map = self._map
        self.final_full_map= self.get_full_map()

        self.block_positions = []
        for i, block in enumerate(self.blocks):
            self.curr_block = i
            block.reset()

        self.curr_block = 0
    
        self.height = 0
        self.old_height = 0
        # self.num_of_bricks = self.total_bricks
        # self.final_map = None
        self.punish = False
        self._map = self.get_map()
        
        self.step = 0
        

        return self._map

    def get_action_space(self, num_actions=5):
        """
            default: left, right, forward, backward, no move, 
            to do: rotate clockwise, rotate counterclockwise
        """
        return spaces.Discrete(num_actions)
    
    def get_observation_space(self):
        """
            default: 3-D grid
        """
        return spaces.Box(low = 0, high = 255, shape = (self.observation_size, self.observation_size, self.observation_size), dtype = np.uint8)
    
    def get_full_map(self):
        map = np.zeros((self.max_x, self.max_y, self.max_z), dtype = float)
        for block in self.blocks:
            map[block.x, block.y, block.z] = 1

        return map
    
    def get_map(self) -> int:
        #for all of the i s immediately above/below, behind/in front, left/right of the current block, check if there is a block there and mark it.

        #assert (self.observation_size <= self.max_x) and (self.observation_size <= self.max_y) and (self.observation_size <= self.max_z)

        assert self.observation_size % 2 == 1, "observation size should be odd" #observation size should be odd so current block can be in the center
        obs = self.get_full_map()
        
        
        obs_offset = self.observation_size//2
        curr_block = self.blocks[self.curr_block]
        #obs[curr_block.x, curr_block.y, curr_block.z] = 5
        x_start = curr_block.x - obs_offset
        x_end = curr_block.x + obs_offset
        y_start = curr_block.y - obs_offset
        y_end = curr_block.y + obs_offset
        z_start = curr_block.z - obs_offset
        z_end = curr_block.z + obs_offset
        
        if x_start < 0:
            x_append = np.zeros(((0-x_start), obs.shape[1], obs.shape[2]), dtype = float)
            x_append.fill(-1)
            obs = obs[:x_end + 1, :, :]
            obs = np.concatenate([x_append, obs], axis = 0)
        if x_end >= self.max_x:
            x_append = np.zeros((1+x_end - self.max_x,  obs.shape[1], obs.shape[2]), dtype = float)
            x_append.fill(-1)
            if x_start >= 0:
                obs = obs[x_start:, :, :]
            obs = np.concatenate([obs, x_append], axis = 0)
        if x_start >= 0 and x_end < self.max_x:
            obs = obs[x_start:x_end+1,:,:]

        if y_start < 0:
            y_append = np.zeros((obs.shape[0], (0-y_start), obs.shape[2]), dtype = float)
            y_append.fill(-1)
            obs = obs[:, :y_end+1, :]
            obs = np.concatenate([y_append, obs], axis = 1)
        if y_end >= self.max_y:
            y_append = np.zeros((obs.shape[0], (1+y_end - self.max_y), obs.shape[2]), dtype = float)
            y_append.fill(-1)
            if y_start >= 0:
                obs = obs[:, y_start:, :]
            obs = np.concatenate([obs, y_append], axis = 1)
        if y_start >= 0 and y_end < self.max_y:
            obs = obs[:,y_start:y_end+1,:]

        if z_start < 0:
            z_append = np.zeros((obs.shape[0], obs.shape[1], (0-z_start)), dtype = float)
            z_append.fill(-1)
            obs = obs[:, :, :z_end+1]
            obs = np.concatenate([z_append, obs], axis = 2)
        if z_end >= self.max_z: 
            z_append = np.zeros((obs.shape[0], obs.shape[1], 1+z_end - self.max_z), dtype = float)
            z_append.fill(-1)
            if z_start >= 0:
                obs = obs[:, :, z_start:]
            obs = np.concatenate([obs, z_append], axis = 2)
        if z_start >= 0 and z_end < self.max_z:
            obs = obs[:,:,z_start:z_end+1]

        assert obs.shape == (self.observation_size, self.observation_size, self.observation_size)

        return obs
        #TODO: map is cubeoid (5 x 5 or 7 x 7) of the space around the current block

    def get_observation(self):
        obs =  (self._last_map + 1)*1/2
        return obs
    
    def update(self, action):
        """
            Take action, determine whether to reward or punish the action. 
        """
        self._set_next_state(self.curr_block, action)

        #if there is a block there, move on top of it
        while self._is_duplicate_state() and self.blocks[self.curr_block].next_x < self.max_x-1:
            self.blocks[self.curr_block].next_x += 1
        if self._is_duplicate_state():
            self.punish = True
            self.old_height = self.height
            self.height = self.get_height()
            self.curr_block = (self.curr_block + 1) % self.num_of_blocks
            self._last_map = self._map
            self._map = self.get_map()
            self.step += 1
            return
        
        #if there is no block underneath, move down
        connected = self._is_connected()
        while (not connected and (self.blocks[self.curr_block].next_x > 0)):
            self.blocks[self.curr_block].next_x -= 1


        self.punish = False

        if self._is_move_valid(self.curr_block, action):
            self._move_to_state(self.curr_block)
        else:
            self.punish = True
        self.old_height = self.height
        self.height = self.get_height()
        self.curr_block = (self.curr_block + 1) % self.num_of_blocks
        self._last_map = self._map
        self._map = self.get_map()
        self.step+=1

        

        # return 1 

    def _is_connected(self):
        curr_block = self.blocks[self.curr_block]
        for i, block in enumerate(self.blocks):
            if i == self.curr_block:
                continue
            elif block.x == curr_block.next_x-1 and block.y == curr_block.next_y and block.z == curr_block.next_z:
                return True
        return False


    def _set_next_state(self, block_num, action):
        """
            Check whether grid has enough space to accommodate 
            the predicted block or not.
        """
        # none = 0, left = 1 , right= 2, forward = 3, backward = 4
        if action == 0:
            pass
        elif action == 1:
            self.blocks[block_num].next_z = max(self.blocks[block_num].z - 1, 0)
        elif action == 2:
            self.blocks[block_num].next_z = min(self.blocks[block_num].z + 1, self.max_z-1)
        elif action == 3:
            self.blocks[block_num].next_y = max(self.blocks[block_num].y - 1, 0)
        elif action == 4:
            self.blocks[block_num].next_y = min(self.blocks[block_num].y + 1, self.max_y-1)
        else:
            print("Invalid Action!!")


    def _move_to_state(self, block_num):
        self.blocks[block_num].x = self.blocks[block_num].next_x
        self.blocks[block_num].y = self.blocks[block_num].next_y
        self.blocks[block_num].z = self.blocks[block_num].next_z

    def _is_move_valid(self, block_num, action):
        """
            Current state should not be equal to old state 
            Current state should be connected
            Current state should not be the same as any other block
        """
        #if self._is_connected(self, block_num, action) == False:
        #    return False
        if action == 0: #take no action
            return True
        if self._is_old_state(block_num) == True:
            return False
        return True
    

    def _is_duplicate_state(self):
        #iterate through the other blocks, see if one is in same spot as moved to. if yes, move on top of it.
        curr_block = self.blocks[self.curr_block]
        for i, block in enumerate(self.blocks):
            if i == self.curr_block:
                continue
            elif block.x == curr_block.next_x and block.y == curr_block.next_y and block.z == curr_block.next_z:
                return True
        return False
        #returns False if there is no duplicate block, and True if there is

    def _is_old_state(self, block_num):
        if self.blocks[block_num].x == self.blocks[block_num].next_x and self.blocks[block_num].y == self.blocks[block_num].next_y and self.blocks[block_num].z == self.blocks[block_num].next_z:
            return True
        else: 
            return False
        
    def get_height(self):
        return max([self.blocks[i].x for i in range(self.num_of_blocks)])#/self.num_of_blocks

    # def get_state_number(self):

    #     return self.y * 100 + self.x * 10 + self.z

