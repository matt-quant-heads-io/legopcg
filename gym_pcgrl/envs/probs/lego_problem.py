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
        map = new_stats["map"]


        if punish:
            reward = -0.5 
        else:
            # reward for decreasing distance from top 
            reward += 1 - ((9 - y)/9)**0.4

            if y > 0:
                # if between levels 1 to 8
                if self._are_studs_connected(map, (y,x,z)):
                    reward += 1
                else:
                    reward -= 2

        self.total_reward += reward

        return reward
    

    def _are_studs_connected(self, map, location):
        """
            call this function only if the agent is in level 1 to 9 
            for a single orientation A brick is connected if and only if 
            it is placed on top of each other
        """
        y,x,z = location

        i = y - 1 

        while i >= 0:
            if map[i][x][z] == 0:
                return False
            i -= 1 

        return True
    
    def get_episode_over(self, new_stats, old_stats):    
        
        if new_stats['num_of_bricks'] <= 0:
            # print("episode over: ", representation.num_of_bricks)
            # print("episode over: ", np.count_nonzero(representation._map))
            # print("Total reward: ",  self.total_reward)
            print("Episode Over.")
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