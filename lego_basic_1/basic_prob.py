import os

class BasicProblem:
    def __init__(self):
        pass

    def getBlockTypes(self):
        return ["brick"]

    def reset(self):
        pass

    def getReward(self):
        # Reward actions that minimize intersection between the cuboids while making them touch
        pass

    def getEpisodeOver(self):
        pass

    # Render - write to .dat file
    def render(self):
        pass
