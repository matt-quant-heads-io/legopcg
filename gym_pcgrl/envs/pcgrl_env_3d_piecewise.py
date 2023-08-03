import copy
import numpy as np
import gym 
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.pcgrl_env_3d import LegoPCGEnv3D
import utils.utils as ut
import gc

class LegoPCGEnv3DPiecewise(gym.Env):
    
    def __init__(self, configs, savedir, problem="legopiecewise", representation="piecewise", model = None):

        super().__init__()
    
        # Initialize Problem and Representation modules
        if configs['scheduled_episodes']:
            self.steps_per_episode = None
        else:
            self.steps_per_episode = configs['num_of_blocks'] * configs['reps_per_episode']
        self.prob = PROBLEMS[problem]()
        self.rep = REPRESENTATIONS[representation](configs, savedir)
       
        # Define Action Space
        self.observation_space = self.rep.get_observation_space()
        self.action_space = self.rep.get_action_space()

        # Number of iterations
        self._episode = 0
        self.reward_param = configs['reward_param']
        self.num_of_blocks = configs['num_of_blocks']
        self._step = 0
        self.savedir = savedir

        self.reward_history = [0]
        self.current_blocks = []
        self.last_blocks = []
        self.model=model

    """
    def seed(self, seed=None):
        seed = self.rep.seed(seed)
        self.prob.seed(seed)
        # return [seed]
    """ 
    def reset(self):
        #if self._iterations_total % 10 == 0 and self._iterations_total > 0:
        #    ut.save_arrangement(self.rep.blocks, self.savedir, self._iterations_total, ut.LegoDimsDict, self.reward_history[-1])

        self.reward_history.append(self.rep.get_reward())
    
        if self._episode % 250 == 0 or self.reward_history[-1] >= max(self.reward_history[:-1]) and self._episode > 20 and self.reward_history[-1] > 0:
            ut.save_arrangement(self.rep.blocks, self.savedir + "training_imgs" + "/", self._episode, self.reward_history[-1], rewards = self.reward_history, render = True, goal = self.rep.last_goal)   
            for i, blocks in enumerate(self.current_blocks):
                ut.save_arrangement(blocks, self.savedir + "training_imgs/" + str(self._episode) +"/", i, curr_reward = None, render = True, goal = self.rep.last_goal)
                if i == len(self.current_blocks)-1:
                    ut.animate(self.savedir + "training_imgs/", self._episode)

            if self.model != None and self.reward_history[-1] >= max(self.reward_history[:-1]):
                self.model.model.save(self.model.saved_model_path + str(self._episode))
        

        """
        if len(self.reward_history) >1 and self.reward_history[-2]-self.reward_history[-1] >= 2:
            for i, blocks in enumerate(self.last_blocks):
                ut.save_arrangement(blocks, self.savedir + "training_imgs/" + str(self._episode-1)+"/", i, curr_reward = None, render = True)  
            if len(self.last_blocks) > 0:
                ut.animate(self.savedir + "training_imgs/", self._episode-1)
            
            for i, blocks in enumerate(self.current_blocks):
                ut.save_arrangement(blocks, self.savedir + "training_imgs/" + str(self._episode) +"/", i, curr_reward = None, render = True)
                if i == len(self.current_blocks)-1:
                    ut.animate(self.savedir + "training_imgs/", self._episode)
        """    
        """
        for i, blocks in enumerate(self.current_blocks):
            ut.save_arrangement(blocks, self.savedir + "training_imgs/" + str(self._episode) +"/", i, curr_reward = None, render = True)
            if i == len(self.current_blocks)-1:
                ut.animate(self.savedir + "training_imgs/" + str(self._episode) +"/")
            
        """
        self._episode += 1
        del self.last_blocks
        self.last_blocks = copy.deepcopy(self.current_blocks)
        del self.current_blocks
        self.current_blocks = []

        self.rep.reset()
        #self.prob.reset()
        self._changes = 0
        #self._step = 0
        gc.collect()
        
        obs = self.rep.get_observation()

        return obs
    
    def step(self, action):
        #ut.save_arrangement(self.rep.blocks, self.savedir + str(self._episode) + "/", self._step, self.reward_history[-1], render = True)

        #self._iteration += 1
        #self._iterations_total += 1
        self._step += 1

        # update the current state to the new state based on the taken action
        self.rep.update(action)
        self.current_blocks.append(copy.deepcopy(self.rep.blocks))

        # Get the next state number
        #observation = self.rep.get_observation()
        
        new_stats = {}
        old_stats = {}

        new_stats['block_num'] = self.rep.curr_block
        old_stats[self.reward_param] = self.rep.last_reward
        #new_stats['punish'] = self.rep.punish
        new_stats[self.reward_param] = self.rep.curr_reward
        new_stats['step'] = self.rep.step
        
        #old_stats['old_location'] = self.rep.old_location
        reward = self.prob.get_reward(new_stats, old_stats, self.reward_param)
        

        #reward = max(reward, 0)

        done = self.prob.get_episode_over(new_stats, self._episode, self.num_of_blocks, self.steps_per_episode)

        

        info = {}
        info['solved'] = done

        # save the map so that it can be used to write dat file
        #if done:
            #self.reward_history.append(self.rep.height)
            #self.rep.final_map = np.copy(self.rep._map)
            #print("Episode Over: ", reward)#, np.count_nonzero(self.rep._map))
        
        observation = self.rep.get_observation()
        #print(self._step)
    
        return observation, reward, done, info
        

    def render(self, mode="lego"):
        pass

        
