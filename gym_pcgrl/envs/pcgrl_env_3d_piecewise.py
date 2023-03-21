import numpy as np
import gym 
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.pcgrl_env_3d import LegoPCGEnv3D
import utils.utils as ut

class LegoPCGEnv3DPiecewise(gym.Env):
    
    def __init__(self, configs, problem="legopiecewise", representation="piecewise"):

        super().__init__()
    
        # Initialize Problem and Representation modules
        self.steps_per_episode = configs['num_of_blocks'] * configs['reps_per_episode']
        self.prob = PROBLEMS[problem]()
        self.rep = REPRESENTATIONS[representation](configs)
       
        # Define Action Space
        self.observation_space = self.rep.get_observation_space()
        self.action_space = self.rep.get_action_space()

        # Number of iterations
        self._iteration = 0
        self.reward_param = configs['reward_param']
        self.num_of_blocks = configs['num_of_blocks']
        self._iterations_total  = 0

        self.reward_history = []

    """
    def seed(self, seed=None):
        seed = self.rep.seed(seed)
        self.prob.seed(seed)
        # return [seed]
    """ 
    def reset(self):
        self.rep.reset()
        #self.prob.reset()
        self._changes = 0
        self._iteration = 0
        if self._iterations_total % 100 == 0 and self._iterations_total > 0:
            ut.save_map_piecewise(self.rep.final_full_map, "/home/maria/dev/legopcg/animations", self._iterations_total, ut.LegoDimsDict, self.reward_history[-1])

        return self.rep.get_observation()

    def step(self, action):

        self._iteration += 1
        self._iterations_total += 1

        # update the current state to the new state based on the taken action
        self.rep.update(action)

        # Get the next state number
        #observation = self.rep.get_observation()
        
        new_stats = {}
        old_stats = {}

        new_stats['block_num'] = self.rep.curr_block
        old_stats['height'] = self.rep.old_height
        new_stats['punish'] = self.rep.punish
        new_stats['height'] = self.rep.height
        new_stats['step'] = self.rep.step
        
        #old_stats['old_location'] = self.rep.old_location
        reward = self.prob.get_reward(new_stats, old_stats, self.reward_param)
        
        done = self.prob.get_episode_over(new_stats, self.steps_per_episode)

        info = {}
        info['solved'] = done

        # save the map so that it can be used to write dat file
        if done:
            self.reward_history.append(self.rep.height)
            #self.rep.final_map = np.copy(self.rep._map)
            #print("Episode Over: ", reward)#, np.count_nonzero(self.rep._map))
        
        observation = self.rep.get_observation()
        return observation, reward, done, info
        

    def render(self, mode="lego"):
        pass

        
