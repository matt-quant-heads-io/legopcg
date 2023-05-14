
import random
import numpy as np
from gym import spaces
from utils.utils import LegoDimsDict
import utils.utils as ut
import sys, os
import pdb

# local imports
from gym_pcgrl.envs.reps.representation_3d import Representation3D 


class LegoBlock():
    def __init__(self, dimsNum, rep):
        super().__init__()
        self.block_name = dimsNum
        self.block_num = ut.str_to_onehot_index_map[dimsNum]
        self.dims = LegoDimsDict[dimsNum]
        self.x = random.randrange(self.dims[0]-1, rep.max_x-self.dims[0]+1)
        self.y = 0
        self.z = random.randrange(self.dims[2]-1, rep.max_z-self.dims[2]+1)
        self.rep = rep
        self.rotation = None #add this as rotation matrix
        self.doesnotfit = False
        self.is_curr_block = False
        self.is_next_block = False
        self.error = None
        curr_positions = []
        for i in range(self.dims[0]):
            for j in range(self.dims[0]):
                for k in range(self.dims[0]):
                    position = str(self.x + i) + str(self.y + j) + str(self.z + k)
                    curr_positions.append(position)

        intersect = [pos for pos in curr_positions if pos in self.rep.block_positions]
        
        ctr = 0
        while len(intersect) > 0 & ctr < 1000:
            ctr+=1
            self.x = random.randrange(self.dims[0]-1, rep.max_x-self.dims[0]+1)
            self.z = random.randrange(self.dims[2]-1, rep.max_z-self.dims[2]+1)

            curr_positions = []
            for i in range(self.dims[0]):
                for j in range(self.dims[1]):
                    for k in range(self.dims[2]):
                        position = str(self.x + i) + str(self.y + j) + str(self.z + k)
                        curr_positions.append(position)
            intersect = [pos for pos in curr_positions if pos in self.rep.block_positions]

        self.next_x = self.x
        self.next_y = self.y
        self.next_z = self.z

        self.last_x = self.x
        self.last_y = self.y
        self.last_z = self.z

        if len(intersect) > 0:
            self.doesnotfit = True
        else:
            rep.block_positions += list(set(curr_positions))
        

    def reset(self):

        self.x = random.randrange(self.dims[0]-1, self.rep.max_x-self.dims[0]+1)
        self.y = 0
        self.z = random.randrange(self.dims[2]-1, self.rep.max_z-self.dims[2]+1)
        self.error = None

        curr_positions = []
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    position = str(self.x + i) + str(self.y + j) + str(self.z + k)
                    curr_positions.append(position)


        ctr = 0
        while len([pos for pos in curr_positions if pos in self.rep.block_positions]) > 0 and ctr < 1000:
            ctr += 1
            self.x = random.randrange(self.dims[0]-1, self.rep.max_x-self.dims[0]+1)
            self.z = random.randrange(self.dims[2]-1, self.rep.max_z-self.dims[2]+1)

            curr_positions = []
            for i in range(self.dims[0]):
                for j in range(self.dims[1]):
                    for k in range(self.dims[2]):
                        position = str(self.x + i) + str(self.y + j) + str(self.z + k)
                        curr_positions.append(position)


        self.next_x = self.x
        self.next_y = self.y
        self.next_z = self.z

        self.last_x = self.x
        self.last_y = self.y
        self.last_z = self.z

        if len([pos for pos in curr_positions if pos in self.rep.block_positions]) > 0:
            self.doesnotfit = True
        else:
            self.rep.block_positions += list(set(curr_positions))

class PiecewiseRepresentation(Representation3D):
    def __init__(self, train_cfg, savedir):
        # To keep count of number of bricks currently in the grid space
        # self.num_of_bricks = 100
        # self.num_of_bricks = total_bricks
        super().__init__()
        
        #self.punish: bool = False
        self.full_map = None
        self.observation_size = train_cfg["observation_size"]
        self.savedir = savedir
        self.punish = train_cfg["punish"]
        self.punish_sum = 0
        self.reward_param = train_cfg["reward_param"]
        self.last_reward = 0
        self.curr_reward = 0

        self.max_x= train_cfg["GridDimensions"][0]# max y starting from 0
        self.max_y = train_cfg["GridDimensions"][1]  # max x starting from 0
        self.max_z = train_cfg["GridDimensions"][2]  # max z starting from 0
        
        self.curr_block = 0
        self.num_of_blocks = train_cfg["num_of_blocks"]
        assert self.num_of_blocks <= self.max_x  * self.max_z

        self.blocks = []
        self.block_positions = []

        for i in range(self.num_of_blocks):
            #TODO: make sure all the blocks will fit
            self.curr_block = i
            if i < self.num_of_blocks//3:
                blockname = "3003"
            elif i < 2*self.num_of_blocks//3:
                blockname = "3004"
            else: 
                blockname = "3005"
            block = LegoBlock(blockname, self)
            if block.doesnotfit:
                pass
            else:
                self.blocks.append(block)
        self.num_of_blocks = len(self.blocks)
        self.curr_block = 0
        self.blocks[0].is_curr_block = True

        #self.full_map= self.get_full_map()
        self._map = self.get_map()
        self._last_map = None
        self.step = 0
        self.step_count = 0
        self.episode = 0
    

        

        
    def reset(self):
        # Reset all internal data structures being used
        # self.final_map = np.copy(self._map)
        #super().reset(height, width, depth)
        self.last_reward = self.curr_reward
        self.curr_reward = self.get_reward()
        self._last_map = np.copy(self._map)
        self.full_map= self.get_full_map()

        self.block_positions = []
        for i, block in enumerate(self.blocks):
            self.curr_block = i
            block.reset()
            if block.doesnotfit:
                self.blocks = self.blocks[:i] + self.blocks[i+1:]

        self.num_of_blocks = len(self.blocks)
        self.curr_block = 0
        self.punish_sum = 0
        self._map = self.get_map()
    
        self.step = 0
        self.step_count = 0
        self.episode += 1

        
        

        return self._map

    def get_action_space(self):
        """
            default: left, right, forward, backward, no move, 
            to do: rotate clockwise, rotate counterclockwise
        """
        #return spaces.MultiDiscrete(nvec=[2*(self.max_x-1), 2*(self.max_y-1), 2*(self.max_z-1)])
        return spaces.MultiDiscrete(nvec=[2*(self.max_x-1), 2*(self.max_y-1)])
    
    def get_observation_space(self):
        """
            default: 3-D grid
        """

        map_obs = spaces.Box(low = -2, high = 5, shape = (self.observation_size, self.observation_size, self.observation_size, 8), dtype = np.uint8)
        self_obs = spaces.Box(low = 0, high = 1, shape = (1,3), dtype = np.uint8)

        return spaces.Dict(
            {
                "block_num": self_obs,
                "map": map_obs,
            }
        )
        #return spaces.Box(low = 0, high = 1, shape = (self.observation_size, self.observation_size, self.observation_size, 8), dtype = np.uint8)
    
    def get_full_map(self):
        map = np.zeros((self.max_x, self.max_y, self.max_z), dtype = float)
        for block in self.blocks:
            #TODO: add rotation component
            for i in range(block.dims[0]):
                for j in range(block.dims[1]):
                    for k in range(block.dims[2]):
                        map[block.x + i, block.y + j, block.z + k] = block.block_num

        curr_block = self.blocks[self.curr_block]
        
        #TODO: add rotation component
        for i in range(curr_block.dims[0]):
            for j in range(curr_block.dims[1]):
                for k in range(curr_block.dims[2]):
                    map[curr_block.x + i, curr_block.y + j, curr_block.z + k] = -2


        return map
    
    def get_map(self) -> int:

        #for all of the i s immediately above/below, behind/in front, left/right of the current block, check if there is a block there and mark it.

        #assert (self.observation_size <= self.max_x) and (self.observation_size <= self.max_y) and (self.observation_size <= self.max_z)

        assert self.observation_size % 2 == 1, "observation size should be odd" #observation size should be odd so current block can be in the center
        obs = self.get_full_map()

        curr_block = self.blocks[self.curr_block]
        for i in range(curr_block.dims[0]):
                for j in range(curr_block.dims[1]):
                    for k in range(curr_block.dims[2]):
                        obs[curr_block.x + i, curr_block.y + j, curr_block.z + k] = -2
        
        
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

        #return obs

        obs = obs +2
        final_obs = np.eye(8)[obs.astype(int)]#np.zeros((obs.shape[0], obs.shape[1], obs.shape[2], 8), type = float)
        return final_obs

    def get_observation(self):
        return {
            "map":self._last_map,
            "block_num":self.blocks[self.curr_block].dims
            }
    
    def _end_update(self):
            #self.punish = True
            
            for block in self.blocks:
                block.next_x = block.x
                block.next_y = block.y
                block.next_z = block.z
            
            self.curr_block = (self.curr_block + 1) % self.num_of_blocks
            self._last_map = self._map
            self._map = self.get_map()
            self.full_map = self.get_full_map()
            #print("ending update ", self.step)

            return
    
    def update(self, action):
        
        """
            Take action, determine whether to reward or punish the action. 
        """                                  
        self.step += 1
        self.punish_sum = 0

        for block in self.blocks:
            block.is_curr_block = False
            block.is_next_block = False
            block.error = None
        self.blocks[self.curr_block].is_curr_block= True

        next_block = (self.curr_block + 1) % self.num_of_blocks
        self.blocks[next_block].is_next_block = True
        
        self.full_map = self.get_full_map()
        #TODO: add rotation component
        self._set_next_state(action)
        
        if self.blocks[self.curr_block].next_x == self.blocks[self.curr_block].x and self.blocks[self.curr_block].next_y == self.blocks[self.curr_block].y:# and self.blocks[self.curr_block].next_z == self.blocks[self.curr_block].z:
            self.punish_sum += float(1)/self.num_of_blocks
            self.blocks[self.curr_block].error = "stay" #orange
            self._end_update()

        if not self._is_in_bounds(self.curr_block):
            self.punish_sum += float(1)/self.num_of_blocks
            self.blocks[self.curr_block].error = "bounds" #purple
            self._end_update()
            return

        if self._is_overlap():
            self.punish_sum -= float(1)/self.num_of_blocks
            self.blocks[self.curr_block].error = "overlap" #blue
            self._end_update()
            return

        #if there is no block underneath, move down
        while (not self._is_connected(self.curr_block)):
            self.blocks[self.curr_block].next_y -= 1
            self._move_to_state(self.curr_block)
            
        self._move_to_state(self.curr_block)
            
        while not self._all_connected():
            self._all_blocks_fall()

            
        if not self._all_connected():
            print("not all connected line 365")
            quit()
          
        self._end_update()
        return
        
    def _all_blocks_fall(self):
        while not self._all_connected():
            self.full_map = self.get_full_map()
            for block_num in range(len(self.blocks)):
                while (not self._is_connected(block_num)):
                    self.full_map = self.get_full_map()
                    self.blocks[block_num].y -= 1
                            

    def _all_connected(self):
        for block_num in range(len(self.blocks)):
                if not self._is_connected(block_num):
                    return False
        return True


    def _is_connected(self, block_num):
        curr_block = self.blocks[block_num]
        if curr_block.y == 0:
            return True

        for x_offset in range(curr_block.dims[0]):
            for z_offset in range(curr_block.dims[2]):
                for y_offset in range(curr_block.dims[1]):
                    if self.get_full_map()[curr_block.x + x_offset,curr_block.y + y_offset-1,curr_block.z + z_offset] != 0:
                        #print("is connected true: ", block_num)
                        #print("pos: ", curr_block.x, curr_block.y, curr_block.z)
                        return True
                    
        #print("is connected false: ", block_num)
        #print("pos: ", curr_block.x, curr_block.y, curr_block.z)
        return False
    

    def get_reward(self):
        if self.reward_param == "avg_height":
            reward = self.punish_sum + self.get_height()
        elif self.reward_param == "avg_height_squared":
            reward = sum([block.y*block.y for block in self.blocks])/len(self.blocks) + self.punish_sum
        if self.punish:
            reward -= self.punish_sum
        return reward
        

    def _set_next_state(self, action):
        """
            Check whether grid has enough space to accommodate 
            the predicted block or not.
        """
        # none = 0, left = 1 , right= 2, forward = 3, backward = 4
        #TO DO : do we punish if it's off the grid
        curr_block = self.blocks[self.curr_block]

        curr_block.next_x += action[0]-self.max_x-1
        curr_block.next_y += action[1]-self.max_y-1
        #curr_block.next_z += action[2]-self.max_z-1
        curr_block.next_z = self.max_z-1

    def _move_to_state(self, block_num):
        curr_block = self.blocks[block_num]
        curr_block.last_x = curr_block.x
        curr_block.last_y = curr_block.y
        curr_block.last_z = curr_block.z

        curr_block.x = curr_block.next_x
        curr_block.y = curr_block.next_y
        curr_block.z = curr_block.next_z


    def _is_in_bounds(self, block_num):
        curr_block = self.blocks[block_num]


        for x_offset in range(curr_block.dims[0]):
            for y_offset in range(curr_block.dims[1]):
                for z_offset in range(curr_block.dims[2]):
                    if curr_block.next_x + x_offset >=self.max_x or curr_block.next_y + y_offset >= self.max_y or curr_block.next_z + z_offset >= self.max_z:
                        return False
                    if curr_block.next_x + x_offset <0 or curr_block.next_y + y_offset < 0 or curr_block.next_z + z_offset <0:
                        return False
        return True


    def _is_overlap(self):

        curr_block = self.blocks[self.curr_block]
        for x_offset in range(curr_block.dims[0]):
            for y_offset in range(curr_block.dims[1]):
                for z_offset in range(curr_block.dims[2]):
                    pos = self.full_map[curr_block.next_x + x_offset,  curr_block.next_y + y_offset,  curr_block.next_z + z_offset]
                    if  pos > 0:
                        return True
        return False
        #returns False if there is no duplicate block, and True if there is

    def _is_old_state(self, block_num):
        if self.blocks[block_num].x == self.blocks[block_num].next_x and self.blocks[block_num].y == self.blocks[block_num].next_y and self.blocks[block_num].z == self.blocks[block_num].next_z:
            return True
        else: 
            return False
        
    def get_height(self):
        return sum([self.blocks[i].y for i in range(self.num_of_blocks)])/self.num_of_blocks

    # def get_state_number(self):

    #     return self.y * 100 + self.x * 10 + self.z

