import numpy as np
# # one hot encoding structure 10 x 10 x 100 x 4

# [[[0,0,1,0], [1,0,0,0] ... [1,0,0,0]]
# .
# .
# .]]]

# # char structure 10 x 10 x 1
# [['wewwwwwww', ...  ...] ...]

# TODO: need following follow maps/methods (dicts): str_to_char_map, 
# onehot_to_char_map 

onehot_index_to_onehot_map = {
    0 : [0,0,0,0],    
    1 : [0,0,0,1],
    2: [0,0,1,0],
    3: [0,1,0,0]
}

str_to_char_map = { 
    'empty': 'w',    
    '3005': 'a',
    '3004': 'b',
    '3622': 'c'
}

char_to_str_map = {
    'a' : '3005',
    'b' : '3004',
    'c' : '3622',
    'w': 'empty'
}

def getBlockName(char):
    return char_to_str_map[char]

str_to_onehot_index_map = { 
    'empty': 0,    
    '3005': 1,
    '3004': 2,
    '3622': 3
}

onehot_index_to_str_map = { 
    0 : 'empty',    
    1 : '3005',
    2: '3004',
    3: '3622'
}

onehot_index_to_str_map = {
    v:k for k,v in str_to_onehot_index_map.items()
}

onehot_index_to_char_map = { 
    0 : 'w',
    1 : 'a',
    2 : 'b',
    3 : 'c'
}

def convertOneHotToChar(oneHotMap):
    charMap = []

    for z in range(len(oneHotMap)):
        for y in range(len(oneHotMap[0])):
            for x in range(len(oneHotMap[0][0])):
                char_tile = onehot_index_to_char_map[oneHotMap[z][y][x]]
                charMap.append(char_tile)

    return np.array(charMap).reshape((10,10,10))
    
def convertOneHotToString(onehot):
    str_map = []
    # onehot = np.array(onehot)
    # print(f"This is a one hot {onehot}")

    for z in range(len(onehot)):
        for y in range(len(onehot[0])):
            for x in range(len(onehot[0][0])):
                # key = list(onehot[z][y]).index(0)
                key = onehot[z][y][x]

                str_tile = onehot_index_to_str_map[key]
                str_map.append(str_tile)
    # print(str_map)
    npArr = np.array(str_map).reshape((10,10,10))
    # print(f"Numpy array is {npArr}")
    return npArr

def convertStringToOneHot(str_map):
    one_hot_map = []

    for z in range(len(str_map)):
        for y in range(len(str_map[0])):
            for x in range(len(str_map[0][0])):
                str_tile = str_map[z][y][x]
                one_hot_idx = str_to_onehot_index_map[str_tile]
                one_hot_tile = [0]*4
                one_hot_tile[one_hot_idx] = 1
                one_hot_map.append(one_hot_tile)

    return np.array(one_hot_map).reshape((10,10,10,4))

def convertStringToChar(stringMap):
    charMap = []

    for z in range(len(stringMap)):
        for y in range(len(stringMap[0])):
            for x in range(len(stringMap[0][0])):
                char_tile = stringMap[z][y][x]
                charMap.append(char_tile)

    return np.array(charMap).reshape((10,10,10))