import numpy as np


# string structure 10 x 10 x 10
[[["empty", "empty", ... "3005", "empty"] 
["empty", "empty", ... "3005", "empty"]
["empty", "empty", ... "3005", "empty"]
["empty", "empty", ... "3005", "empty"]
["empty", "empty", ... "3005", "empty"]
["empty", "empty", ... "3005", "empty"]
["empty", "empty", ... "3005", "empty"]
["empty", "empty", ... "3005", "empty"]
["empty", "empty", ... "3005", "empty"]
["empty", "empty", ... "3005", "empty"]]  ...]           # "3004", "3622"]

# one hot encoding structure 10 x 10 x 100 x 4

[[[0,0,1,0], [1,0,0,0] ... [1,0,0,0]]
.
.
.]]]


# char structure 10 x 10 x 1
[['wewwwwwww', ...  ...] ...]


# TODO: need following follow maps/methods (dicts): str_to_char_map, 
# onehot_to_char_map 

str_to_onehot_map = {
    'empty': 0,
    '3005': 1,
    '3004': 2,
    '3622': 3
}

onehot_to_str_map = {
    v:k for k,v in str_to_onehot_map.items()
}



def get_string_from_onehot(onehot):
    str_map = []

    for z in range(len(onehot)):
        for y in range(len(onehot[0])):
            for x in range(len(onehot[0][0])):
                key = onehot[z][y][x].index(1) 
                str_tile = onehot_to_str_map[key]
                str_map.append(str_tile)
    return np.array(str_map).reshape((10,10,10))


def str_map_to_onehot(str_map):
    one_hot_map = []

    for z in range(len(str_map)):
        for y in range(len(str_map[0])):
            for x in range(len(str_map[0][0])):
                str_tile = str_map[z][y][x]
                one_hot_idx = str_map_to_onehot[str_tile]
                one_hot_tile = [0]*4
                one_hot_tile[one_hot_idx] = 1
                one_hot_map.append(one_hot_tile)

    return np.array(one_hot_map).reshape((10,10,10,4))
