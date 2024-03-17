import numpy as np
from gym import spaces

# local imports
from gym_envs.gym_pcgrl.envs.reps.representation_3d import Representation3D


class TurtleRepresntation3D(Representation3D):
    def __init__(self):
        super().__init__()
        # self.old_location = []
        # self.old_location = {}
        self.max_height = None  # max y starting from 0
        self.max_width = None  # max x starting from 0
        self.max_depth = None  # max z starting from 0

    def reset(self, **kwargs):
        super().reset(**kwargs)
        height, width, depth = kwargs.get("grid_dimensions")
        self.y = 0
        self.x = 5
        self.z = 5
        self.max_height = height - 1
        self.max_width = width - 1
        self.max_depth = depth - 1

        # return self._map

    def get_action_space(self, **kwargs):
        """
        Actions : left, right, up, down, rise and fall
        Tiles :
        """
        num_actions = 6
        tiles = len(kwargs.get("lego_block_ids"))
        return spaces.Discrete(num_actions + tiles)

    def get_observation_space(self, **kwargs):
        height, width, depth = kwargs.get("crop_dimensions")
        num_tiles = len(kwargs.get("lego_block_ids"))
        return spaces.Box(
            low=0, high=num_tiles, shape=(height, width, depth), dtype=np.uint8
        )

    def get_observation(self):
        return self._map

    def update(self, action: int):
        """
        the update method is called inside the Lego gym environment's step method.
        action is a tuple of multidiscrete and discrete sample
        """
        self.punish = False
        self.brick_added = False
        # print(self.y,self.x,self.z)
        # self.old_location[(self.y,self.x,self.z)] = 0

        if action < 6:
            self._move_to_state(action)
            self.predicted_location = (self.y, self.x, self.z)
            # if action didn't result in a valid movement
            # if self.old_location == [self.y,self.x,self.z]:
            #     self.punish = True
            # if (self.y,self.x,self.z) in self.old_location:
            #     self.punish = True
        else:
            # tiles 0,1,2,3 etc.
            brick_type = action - 6
            tile_x, tile_y, tile_z = self.lego_block_dimensions_dict[
                self.lego_block_ids[brick_type]
            ]
            self.predicted_location = (self.y, self.x, self.z)

            if brick_type > 0:
                if self._is_valid_location(tile_y, tile_x, tile_z):

                    self.num_bricks -= 1
                    self.brick_added = True
                    self.brick_details.append((self.y, self.x, self.z, brick_type))

                    self.render_map[self.y][self.x][self.z] = brick_type

                    for i in range(0, tile_y):
                        for j in range(0, tile_x):
                            for k in range(0, tile_z):
                                self._map[self.y + i][self.x + j][
                                    self.z + k
                                ] = brick_type
                                # map the filled locations to original location where brick is being placed
                                # self.brick_locations[(self.y+i,self.x+j,self.z+k)] = (self.y,self.x,self.z)

                    self.x += tile_x
                else:
                    self.punish = True
            else:
                pass  # agent has chosen to not place a brick here

    def _move_to_state(self, action):

        """
        Check whether grid has enough space to accommodate
        the predicted block or not.
        """
        # left = 0 , right= 1, up = 2, down = 3, rise = 4, fall = 5
        if action == 0:
            self.x = max(self.x - 1, 0)
        elif action == 1:
            self.x = min(self.x + 1, 9)
        elif action == 2:
            self.z = min(self.z + 1, 9)
        elif action == 3:
            self.z = max(self.z - 1, 0)
        elif action == 4:
            self.y = min(self.y + 1, 9)
        elif action == 5:
            self.y = max(self.y - 1, 0)
        else:
            print("Invalid Action!!")

    # def _is_valid_location(self):

    #     if self._map[self.y][self.x][self.z] != 0:
    #         return False

    #     if self.y - 1 >= 0 and self._map[self.y-1][self.x][self.z] == 0:
    #         return False

    #     # if self.x - 1 >= 0 and self._map[self.y][self.x-1][self.z] == 0:
    #     #     return False

    #     # if self.x + 1 < 10 and self._map[self.y][self.x+1][self.z] == 0:
    #     #     return False

    #     # if self.z - 1 >= 0 and self._map[self.y][self.x][self.z - 1] == 0:
    #     #     return False

    #     # if self.z + 1 < 10 and self._map[self.y][self.x][self.z + 1] == 0:
    #     #     return False
    #     #
    #     return True
