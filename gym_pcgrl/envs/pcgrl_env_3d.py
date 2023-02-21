import numpy as np
import gym 
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS

class LegoPCGEnv3D(gym.Env):
    
    def __init__(self, problem="lego3d", representation="turtle3d"):

        super(LegoPCGEnv3D, self).__init__()
    
        # Initialize Problem and Representation modules
        self.prob = PROBLEMS[problem]()
        self.rep = REPRESENTATIONS[representation]()
       
        # Define Action Space
        self.observation_space = self.rep.get_observation_space()
        self.action_space = self.rep.get_action_space()

        # Number of iterations
        self._iteration = 0

    def seed(self, seed=None):
        seed = self.rep.seed(seed)
        self.prob.seed(seed)
        # return [seed]
      
    def reset(self):
        self.rep.reset(self.prob.height, self.prob.width, 
                        self.prob.depth, self.prob.total_bricks)
        self.prob.reset()
        self._changes = 0
        self._iteration = 0
        observation = 0
        return observation

    def step(self, action):

        self._iteration += 1

        # update the current state to the new state based on the taken action
        self.rep.update(action)

        # Get the next state number
        observation = self.rep.get_observation()
        new_stats = {}
        new_stats['new_location'] = [self.rep.y, self.rep.x, self.rep.z]
        new_stats['punish'] = self.rep.punish
        new_stats['num_of_bricks'] = self.rep.num_of_bricks
        old_stats = {}
        old_stats['old_location'] = self.rep.old_location
        reward = self.prob.get_reward(new_stats, old_stats)
        
        done = self.prob.get_episode_over(new_stats, old_stats)

        info = {}
        info['solved'] = done

        # save the map so that it can be used to write dat file
        if done:
            self.rep.final_map = np.copy(self.rep._map)
            print("Episode Over: ", np.count_nonzero(self.rep._map))
        return observation, reward, done, info

    def render(self, mode="lego"):
        pass

        
