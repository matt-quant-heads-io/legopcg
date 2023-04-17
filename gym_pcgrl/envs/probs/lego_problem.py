# package imports
import matplotlib.pyplot as plt 

# local import 
from gym_pcgrl.envs.probs.problem_3d import Problem3D
from gym_pcgrl.envs.helper import get_lego_reward, LegoReward

class LegoProblem(Problem3D):
    """ 
        We define information related to 'Lego building construction' in this class.        
    """
    def __init__(self) -> None:
        super().__init__()
        self.total_reward = 0 
        self.reward_history = []
        self.lego_reward = None

    def reset(self, **kwargs):
        rep_type = kwargs.get('rep_type', None)
        self.lego_reward = LegoReward(rep_type)

    def get_tile_types(self):
        pass 

    def get_stats(self, map):
        pass 

    
    def get_reward(self, new_stats, old_stats):
        
        reward = self.lego_reward.get_reward(new_stats, old_stats)
        self.total_reward += reward
        # # print(self.total_reward)
        return reward
    

    # def _are_studs_connected(self, map, location):
    #     """
    #         call this function only if the agent is in level 1 to 9 
    #         for a single orientation A brick is connected if and only if 
    #         it is placed on top of each other
    #     """
    #     y,x,z = location

    #     i = y - 1 

    #     while i >= 0:
    #         if map[i][x][z] == 0:
    #             return False
    #         i -= 1 

    #     return True
    
    def get_episode_over(self, new_stats, old_stats):    
        
        # y,x,z = new_stats['new_location']
        if new_stats['num_of_bricks'] <= 0:
        # if (y == 9 and x == 9 and z == 9) or new_stats['num_of_bricks'] <= 0:
            # print("episode over: ", representation.num_of_bricks)
            # print("episode over: ", np.count_nonzero(representation._map))
            print("Episode Over.")
            print("Total reward: ",  self.total_reward)
            self.reward_history.append(self.total_reward)
            self.total_reward = 0 
            # self.total_height = 0
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