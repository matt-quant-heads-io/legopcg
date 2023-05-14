# package imports
import math
import matplotlib.pyplot as plt 

# local import 
from gym_pcgrl.envs.probs.lego_problem import LegoProblem



class LegoProblemPiecewise(LegoProblem):
    """ 
        We define information related to 'Lego building construction' in this class.        
    """
    def __init__(self) -> None:
        super().__init__()
        self.total_reward = 0 
        self.reward_history = []

    def get_tile_types(self):
        pass 

    def get_stats(self, map):
        pass 

    def get_reward(self, new_stats, old_stats, reward_param):
        reward = 0
        #punish = new_stats['punish']

        # best reward condition so far 
        #if punish:
        #    reward = -0.5
        #else:
        #    reward = new_stats[reward_param] - old_stats[reward_param]

        reward = new_stats[reward_param] - old_stats[reward_param]
        # Print Reward graph -> Accumulate rewards
        # print("Reward: ", reward)
        self.total_reward += reward

        return reward
    
    def get_episode_over(self, new_stats, episode, num_blocks, steps_per_ep = None):    
        
        if steps_per_ep == None:
            steps_per_ep = math.sqrt(episode) * num_blocks
            #steps_per_ep = episode//4
    
        #print(steps_per_ep)

        if new_stats['step'] >=  steps_per_ep:
            # print("episode over: ", representation.num_of_bricks)
            # print("episode over: ", np.count_nonzero(representation._map))
            # print("Total reward: ",  self.total_reward)
            self.reward_history.append(self.total_reward)
            self.total_reward = 0 
            return True
        
        return False

    def get_debug_info(self, new_stats, old_stats):
        pass 

    
    def plot_reward(self):
        plt.plot(range(len(self.reward_history)), self.reward_history)
        plt.title('Episodes - Rewards Plot')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig('rewards.png')

    def reset(self):
        pass
        #TODO: what needs to happen here?