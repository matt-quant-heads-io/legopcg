# package imports
import matplotlib.pyplot as plt 

# local import 
from gym_pcgrl.envs.probs.problem_3d import Problem3D


class LegoProblem(Problem3D):
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

    def get_reward(self, new_stats, old_stats):
        reward = 0
        y, x, z = new_stats['new_location']
        punish = new_stats['punish']
        old_y = old_stats['old_location'][0]

        # best reward condition so far 
        if (y > old_y or 
            (y == 9)):
            if punish:
                reward = -0.5
            else:
                reward = 2
        else:
            reward = -1

        # Print Reward graph -> Accumulate rewards
        # print("Reward: ", reward)
        self.total_reward += reward

        return reward
    
    def get_episode_over(self, new_stats, old_stats):    
        
        if new_stats['num_of_bricks'] <= 0:
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