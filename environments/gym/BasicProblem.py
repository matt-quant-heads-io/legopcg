import os
import numpy as np
from PIL import Image
import random

class BasicProblem:
    def __init__(self):
        
        # Create a 10x10x10 grid - Can be generalized later
        self._length = 10
        self._width = 10
        self._height = 10

    def get_tile_types(self):
        return ["empty", "3005", "3004", "3622"]

    def get_stats(self, rep):
        return {
            "connected_studs": self.computeNumberOfConnectedStuds(rep._isStudConnected),
            "render-error": rep._isRenderError
        }

    def computeNumberOfConnectedStuds(self, isStudConnectedMap):
        sum = 0
        for y in range(len(isStudConnectedMap[0])):
            for z in range(len(isStudConnectedMap)):
                for x in range(len(isStudConnectedMap[0][0])):
                    sum += isStudConnectedMap[z][y][x]

        return sum

    def get_reward(self, new_stats, old_stats):
        
        reward = 0

        if new_stats["connected_studs"] > old_stats["connected_studs"]:
            reward += 1
        elif new_stats["connected_studs"] < old_stats["connected_studs"]:
            reward -= 1
        
        if new_stats["render-error"]:
            reward -= 1

        return reward

    def get_episode_over(self, new_stats):
        return new_stats["connected_studs"] >= 500

