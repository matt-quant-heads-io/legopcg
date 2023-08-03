
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
    def __init__(self, dimsNum, rep, block_idx):
        super().__init__()
        self.block_name = dimsNum
        self.block_num = ut.str_to_onehot_index_map[dimsNum]
        self.dims = LegoDimsDict[dimsNum]
        self.idx = block_idx

        self.rep = rep
        self.rotation = None #add this as rotation matrix

        self.is_curr_block = False
        self.is_next_block = False
        self.error = None
        self.x = random.randrange(self.dims[0]-1, rep.max_x-self.dims[0]+1)
        self.y = self.rep.max_y-self.dims[1]
        self.z = random.randrange(self.dims[2]-1, rep.max_z-self.dims[2]+1)

        self.next_x = self.x
        self.next_y = self.y
        self.next_z = self.z

        self.last_x = self.x
        self.last_y = self.y
        self.last_z = self.z
        

    def place(self):
        self.rep.curr_block = self.idx

        #random x random z, top of board
        while self.rep._is_overlap():
            self.x = random.randrange(self.dims[0]-1, self.rep.max_x-self.dims[0]+1)
            self.y = self.rep.max_y-self.dims[1]
            self.z = random.randrange(self.dims[2]-1, self.rep.max_z-self.dims[2]+1)

            self.next_x = self.x
            self.next_y = self.y
            self.next_z = self.z

            self.last_x = self.x
            self.last_y = self.y
            self.last_z = self.z

        #fall to floor
        self._fall()

        
        
    def _fall(self):

  
        self.rep.full_map = self.rep.get_full_map()
        while not self.rep._is_connected(self.idx):
 
            self.rep.full_map = self.rep.get_full_map()
            self.y -= 1
            self.next_y = self.y

            
        

        

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
        self.action_space = train_cfg["action_space"]
        self.policy = train_cfg["policy"]

        self.max_x= train_cfg["GridDimensions"][0]# max y starting from 0
        self.max_y = train_cfg["GridDimensions"][1]  # max x starting from 0
        self.max_z = train_cfg["GridDimensions"][2]  # max z starting from 0


        self.controllable = train_cfg["controllable"]

        if self.controllable:
            self.goal = [random.randrange(self.max_x), random.randrange(self.max_z)]
            self.last_goal = [x for x in self.goal]

        else:
            self.last_goal = None
        
        self.curr_block = 0
        self.num_of_blocks = train_cfg["num_of_blocks"]
        assert self.num_of_blocks <= self.max_x  * self.max_z

        self.blocks = []
        self.block_positions = []

        for i in range(2):
            block = LegoBlock("3031", self, i)
            self.blocks.append(block)
            block.place()
        
        for i in range(2,self.num_of_blocks):

            blockname = ut.onehot_index_to_str_map[random.randint(1, 3)]
            block = LegoBlock(blockname, self, i)
            self.blocks.append(block)
            block.place()

        self.curr_block = 0
        self.blocks[0].is_curr_block = True

        

        #self.full_map= self.get_full_map()
        self._map = self.get_map()
        self._last_map = np.copy(self._map)
    
        self.step = 0
        self.step_count = 0
        self.episode = 0
    

        

        
    def reset(self):
        # Reset all internal data structures being used


        self.final_map = np.copy(self._map)
        #super().reset(height, width, depth)

        if self.controllable and self.episode%10 == 0:
            self.last_goal = [x for x in self.goal]
            self.goal = [random.randrange(self.max_x), random.randrange(self.max_z)]
        self.full_map= self.get_full_map()

        self.blocks = []
        
        for i in range(2):
            block = LegoBlock("3031", self, i)
            self.blocks.append(block)
            block.place()
        
        for i in range(2,self.num_of_blocks):

            blockname = ut.onehot_index_to_str_map[random.randint(1, 3)]
            """
            if i < self.num_of_blocks//4:
                blockname = "3003"
            elif i < 2*self.num_of_blocks//4:
                blockname = "3004"
            else: 
                blockname = "3005"
            """
            block = LegoBlock(blockname, self, i)
            self.blocks.append(block)
            block.place()



        self.curr_block = 0
        self.punish_sum = 0
        self._map = self.get_map()
        self._last_map = np.copy(self._map)
    
        self.step = 0
        self.step_count = 0
        self.episode += 1
        if self.controllable:
            self.goal = (random.randrange(self.max_x), random.randrange(self.max_z))

        
        

        return self._map

    def get_action_space(self):
        """
            default: left, right, forward, backward, no move, 
            to do: rotate clockwise, rotate counterclockwise
        """
        #return spaces.MultiDiscrete(nvec=[2*(self.max_x-1), 2*(self.max_y-1), 2*(self.max_z-1)])
        if self.action_space == "relative_position":
            return spaces.MultiDiscrete(nvec=[3, 3])#spaces.MultiDiscrete(nvec=[2*(self.max_x)-1, 2*(self.max_z)-1])
        
        elif self.action_space == "one_step":
            return spaces.Discrete(5)
        elif self.action_space == "fixed_position":
            return spaces.MultiDiscrete(nvec=[self.max_x, self.max_z])
        else:
            print("Invalid Action Space!")

    def get_observation_space(self):
        """
            default: 3-D grid
        """

        #map_obs = spaces.Box(low = -2, high = 5, shape = (self.observation_size, self.observation_size, self.observation_size, 8), dtype = np.uint8)
        if self.action_space == "relative_position" or self.action_space == "one_step":
            map_obs = spaces.Box(low = 0, high = 1, shape = (2+len(ut.onehot_index_to_str_map), self.observation_size, self.observation_size*3-2, self.observation_size), dtype = np.uint8)
        elif self.action_space == "fixed_position":
            map_obs = spaces.Box(low = 0, high = 1, shape = (2+len(ut.onehot_index_to_str_map), self.max_x, self.max_y, self.max_z), dtype = np.uint8)

        self_obs = spaces.Box(low = 0, high = 1, shape = (1,3), dtype = np.uint8)

        goal_obs = spaces.Box(low=0, high = max(self.max_x, self.max_z), shape=(1,2), dtype=np.uint8)
        

        if self.controllable:
            return spaces.Dict(
                {
                    "goal": goal_obs,
                    "block_dims": self_obs,
                    "map": map_obs,
                }
            )
        else:
            return spaces.Dict(
                {
                    "block_dims": self_obs,
                    "map": map_obs,
                }
            )
    

    def get_observation(self):
        
        if self._last_map is None:
            self._last_map = self.get_map()

        if self.controllable:
            return {
                "goal": self.goal,
                "map":self._last_map,
                #"position: ?"
                "block_dims":self.blocks[self.curr_block].dims

            }

        else:
            return {
                "map":self._last_map,
                #"position: ?"
                "block_dims":self.blocks[self.curr_block].dims

            }
    
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
    
    def get_relative_map(self):

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
        #obs[curr_block.x, curr_block.yf, curr_block.z] = 5
        x_start = curr_block.x - obs_offset
        x_end = curr_block.x + obs_offset
        y_start = curr_block.y - obs_offset*3
        y_end = curr_block.y + obs_offset*3
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
    
        assert obs.shape == (self.observation_size, self.observation_size*3-2, self.observation_size)

        #use below for one hot
        obs = obs +2
        onehot_obs = np.eye(len(ut.onehot_index_to_str_map)+2)[obs.astype(int)]#np.zeros((obs.shape[0], obs.shape[1], obs.shape[2], len()), type = float)
        
        final_obs =  np.transpose(onehot_obs, (3,0,1,2))#.astype(np.float32)
        return(final_obs)
            
    def get_fixed_map(self):

        #for all of the i s immediately above/below, behind/in front, left/right of the current block, check if there is a block there and mark it.

        #assert (self.observation_size <= self.max_x) and (self.observation_size <= self.max_y) and (self.observation_size <= self.max_z)

        obs = self.get_full_map()

        curr_block = self.blocks[self.curr_block]

        for i in range(curr_block.dims[0]):
                for j in range(curr_block.dims[1]):
                    for k in range(curr_block.dims[2]):
                        obs[curr_block.x + i, curr_block.y + j, curr_block.z + k] = -2
        
        
        obs = obs +2
        onehot_obs = np.eye(len(ut.onehot_index_to_str_map)+2)[obs.astype(int)]#np.zeros((obs.shape[0], obs.shape[1], obs.shape[2], len()), type = float)
        
        final_obs =  np.transpose(onehot_obs, (3,0,1,2))#.astype(np.float32)
        return(final_obs)
    
    def get_map(self):
        if self.action_space == "fixed_position":
            return self.get_fixed_map()
        else:
            return self.get_relative_map()
    
    def _end_update(self):
            #self.punish = True
            self._last_map = np.copy(self._map)
            for block in self.blocks:
                block.next_x = block.x
                block.next_y = block.y
                block.next_z = block.z
            
            self.curr_block = (self.curr_block + 1) % self.num_of_blocks
            self._last_map = np.copy(self._map)
            self._map = self.get_map()
            self.full_map = self.get_full_map()

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
        
        
        if self.blocks[self.curr_block].next_x == self.blocks[self.curr_block].x and self.blocks[self.curr_block].next_z == self.blocks[self.curr_block].z:# and self.blocks[self.curr_block].next_z == self.blocks[self.curr_block].z:
            self.punish_sum += self.punish
            self.blocks[self.curr_block].error = "stay" #orange
            self._end_update()

        if not self._is_in_bounds(self.curr_block):
            self.punish_sum += self.punish
            self.blocks[self.curr_block].error = "bounds" #purple
            self._end_update()
            return

        if self._is_overlap():
            self.punish_sum += self.punish
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

    def _get_covered_volume(self):
        covered_volume = 0
        for curr_block in self.blocks:
            for x_offset in range(curr_block.dims[0]):
                for z_offset in range(curr_block.dims[2]):
                    if self.get_full_map()[curr_block.x + x_offset, curr_block.y-1, curr_block.z + z_offset] == 0 :
                        curr_y = curr_block.y-1
                        while curr_y >= 0:
                            if self.get_full_map()[curr_block.x + x_offset, curr_y, curr_block.z + z_offset] == 0:
                                covered_volume += 1
                                curr_y -=1
                            else:
                                break
        return covered_volume            

    def _is_connected(self, block_num):
        curr_block = self.blocks[block_num]
        if curr_block.y == 0:
            return True

        for x_offset in range(curr_block.dims[0]):
            for z_offset in range(curr_block.dims[2]):
                if self.get_full_map()[curr_block.x + x_offset, curr_block.y-1, curr_block.z + z_offset] != 0 :
                    #print("is connected true: ", block_num)
                    #print("pos: ", curr_block.x, curr_block.y, curr_block.z)
                    return True
                    
        #print("is connected false: ", block_num)
        #print("pos: ", curr_block.x, curr_block.y, curr_block.z)
        return False
    
    
    
    def _will_be_connected(self, block_num):
        curr_block = self.blocks[block_num]
        if curr_block.next_y == 0:
            return True
        
        if curr_block.next_x +curr_block.dims[0]-1 >= self.max_x or curr_block.next_z+curr_block.dims[2]-1 >= self.max_z:
            return True
        else:

            for x_offset in range(curr_block.dims[0]):
                for z_offset in range(curr_block.dims[2]):
                    if self.get_full_map()[curr_block.next_x + x_offset, curr_block.next_y-1, curr_block.next_z + z_offset] != 0 :
                        #print("is connected true: ", block_num)
                        #print("pos: ", curr_block.x, curr_block.y, curr_block.z)
                        return True
                        
            #print("is connected false: ", block_num)
            #print("pos: ", curr_block.x, curr_block.y, curr_block.z)
            return False
    

    def get_reward(self):
        if self.reward_param == "avg_height":
            
            if self.controllable:
                reward=self.height_at_goal()
            else:
                reward = self.get_height()
        elif self.reward_param == "avg_height_squared":
            reward = sum([block.y*block.y for block in self.blocks])/len(self.blocks)
        elif self.reward_param == "platform":
            max_height = max([(block.y + block.dims[1] - 1) for block in self.blocks])
            platform_size = sum([(block.dims[0] *block.dims[2]) for block in self.blocks if (block.y + block.dims[1] - 1) == max_height])
            reward = max_height * platform_size
        elif self.reward_param == "volume_covered":
            reward = self._get_covered_volume()

        else: 
            print("Invalid Reward!")
            quit()
        reward -= self.punish_sum

        return reward
        
    

    def _set_next_state(self, action):
        """
            Check whether grid has enough space to accommodate 
            the predicted block or not.
        """
        
        curr_block = self.blocks[self.curr_block]
        
        if self.action_space == "one_step":
            #none = 0, left = 1 , right= 2, forward = 3, backward = 4
            if action == 0:
                pass
            elif action == 1:
                curr_block.next_z = curr_block.z - 1#max(curr_block.z - 1, 0)
            elif action == 2:
                curr_block.next_z = curr_block.z + 1#min(curr_block.z + 1, self.max_z-curr_block.dims[2])
            elif action == 3:
                curr_block.next_x = curr_block.x - 1#max(curr_block.x - 1, 0)
            elif action == 4:
                curr_block.next_x = curr_block.x + 1#min(curr_block.x + 1, self.max_x-curr_block.dims[0])
            else:
                print("Invalid Action!!")

            curr_block.next_y = self.max_y-curr_block.dims[1]


        elif self.action_space == "relative_position":
            move_x = action[0] - 1
            next_x = curr_block.x + move_x
            move_z = action[1] - 1
            next_z = curr_block.z + move_z

            curr_block.next_x = next_x#max(min(self.max_x-1, next_x), 0)
            curr_block.next_z = next_z#max(min(self.max_z-1, next_z), 0)
            curr_block.next_y = self.max_y-curr_block.dims[1]

            #print(curr_block.next_x, curr_block.next_z)
        
        elif self.action_space == "fixed_position":

            
            curr_block.next_x = action[0]#max(min(self.max_x-1, next_x), 0)
            curr_block.next_z = action[1]#max(min(self.max_z-1, next_z), 0)
            curr_block.next_y = self.max_y-curr_block.dims[1]

            #print(curr_block.next_x, curr_block.next_z)
        else:
            print("Invalid Action Space!")
        """
        # 

        """
        while not self._will_be_connected(self.curr_block):
            curr_block.next_y-=1

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
                        #print("returning false")
                        return False
                    if curr_block.next_x + x_offset <0 or curr_block.next_y + y_offset < 0 or curr_block.next_z + z_offset <0:
                        #print("returning false")
                        return False
        #print("returning true")
        return True


    def _is_overlap(self):

        curr_block = self.blocks[self.curr_block]
        for x_offset in range(curr_block.dims[0]):
            for y_offset in range(curr_block.dims[1]):
                for z_offset in range(curr_block.dims[2]):
                    pos = self.get_full_map()[curr_block.next_x + x_offset,  curr_block.next_y + y_offset,  curr_block.next_z + z_offset]
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
    
    def height_at_goal(self):
        goal_x, goal_z = self.goal

        map = self.get_full_map()
        
        for y in range(self.max_z-1, -1, -1):
            if map[goal_x, y, goal_z] != 0:
                return y
        return 0




    # def get_state_number(self):

    #     return self.y * 100 + self.x * 10 + self.z

