import numpy as np
import os
import sys
import subprocess
import imageio

import re




LegoDimsDict = {
    "empty" : [0,0,0],
    "3005" : [1,1,1], # x,y,z dims
    "3004" : [2,1,1],
    "3622" : [3,1,1],
    "3002" : [3,1,2]
}


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



def createGIF():
    ANIMATION_PATH = 'animations'

    walk_dir = f'{ANIMATION_PATH}/'
    walk_dir = os.path.abspath(walk_dir)
    print('walk_dir = ' + walk_dir)
    # print('walk_dir (absolute) = ' + )

    for root, subdirs, files in os.walk(walk_dir):
        print('--\nroot = ' + root)
        
        for subdir in subdirs:
            print('\t- subdirectory ' + subdir)
        
        if (len(files) != 0):
            files.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        for filename in files:
            if filename.endswith(".dat") and not(filename.startswith("step_0")):
                file_path = os.path.join(root, filename)            

                print('\t- file %s (full path: %s)' % (filename, file_path)) 
                origCwd = os.getcwd()
                os.chdir(root)
                cwd = os.getcwd()
                print("cwd: " + cwd)
                l3pCommand = "L3P.EXE -b -f -cg150,150,450 " + filename
                subprocess.run(l3pCommand, shell=True, check=True)   
                povFileName = filename.split('.')[0] + ".pov"
                povCommand = "pvengine.exe /EXIT /RENDER " + povFileName
                subprocess.run(povCommand, shell=True, check=True)
                os.chdir(origCwd) 

    # Create GIF
    # gifFilePath = walk_dir + "/finalAnimation.gif"
    # with imageio.get_writer(gifFilePath, mode='I',duration=0.5) as writer:
    for root, subdirs, files in os.walk(walk_dir):
        print('--\nroot = ' + root)
        
        for subdir in subdirs:
            print('\t- subdirectory ' + subdir)
        
        # Create GIF
        

        if (len(files) != 0):
            files.sort(key=lambda f: int(re.sub('\D', '', f)))

            gifFileName = root.split("\\")[-1] + ".gif"
            gifFilePath = root + "/" + gifFileName

            with imageio.get_writer(gifFilePath, mode='I',duration=0.5) as writer:
                for filename in files:
                    if filename.endswith(".png"):
                        png_file_path = os.path.join(root, filename)            

                        print('\t- file %s (full path: %s)' % (filename, png_file_path)) 

                        image = imageio.imread(png_file_path)
                        writer.append_data(image)                

def write_curr_obs_to_dir_path(obsOneHot, dir_path, curr_step_num):
    charMap = convertOneHotToChar(obsOneHot)
    f = open(dir_path+'/step_'+str(curr_step_num)+'.txt', "a")
    f.write(str(charMap))
    f.close()
    generateDatFile(charMap, dir_path,curr_step_num)

def generateDatFile(charMap, dir_path,curr_step_num):
    f = open(dir_path+'/step_'+str(curr_step_num)+'.dat', "a")
    f.write("0\n")
    f.write("0 Name: New Model.ldr\n")
    f.write("0 Author:\n")
    f.write("\n")

    
    startBlockChar = charMap[0][0][0]
    startBlockName = char_to_str_map[startBlockChar]
    startBlockDimensions = LegoDimsDict[startBlockName]
    
    for y in range(len(charMap[0])):
        for z in range(len(charMap)):
            for x in range(len(charMap[0][0])):
                char = charMap[z][y][x]

                if (char != 'w'):
                    # Get Dimensions of Lego Block
                    legoBlockName = char_to_str_map[char]
                    currentXYZDims = LegoDimensions.LegoDimsDict[legoBlockName]

                    # Along x-dirn
                    factor = 0
                    if (startBlockDimensions[0] != 0):
                        if (currentXYZDims[0] > startBlockDimensions[0]):
                            factor = (currentXYZDims[0] - startBlockDimensions[0]) * 10
                        elif (currentXYZDims[0] < startBlockDimensions[0] ): # was >0
                            factor = (currentXYZDims[0] - startBlockDimensions[0]) * 10  
                            

                    
                        
                    xLego = x * 20 + factor
                    yLego = y * -24  
                    # yLego = y * 24 * currentXYZDims[1] 
                    zLego = z * 20 

                    

                if (char == 'w'):
                    a=1

                elif (char == 'a'):
                    f.write("1 7 ")
                    
                    f.write(str(xLego) + ' ' + str(yLego) + ' ' + str(zLego) + ' ')
                    f.write("1 0 0 0 1 0 0 0 1 ")
                    f.write(getBlockName(char) + ".dat")
                    f.write("\n")

                elif (char == 'b'):
                    f.write("1 7 ")
                    f.write(str(xLego) + ' ' + str(yLego) + ' ' + str(zLego) + ' ')
                    f.write("1 0 0 0 1 0 0 0 1 ")
                    f.write(getBlockName(char) + ".dat")
                    f.write("\n")

                elif (char == 'c'):
                    f.write("1 7 ")
                    f.write(str(xLego) + ' ' + str(yLego) + ' ' + str(zLego) + ' ')
                    f.write("1 0 0 0 1 0 0 0 1 ")
                    f.write(getBlockName(char) + ".dat")
                    f.write("\n")
                
    f.close()   




    # Used to write interpolated .mpd file (to be called inside write file function)
def interpolate_template_map_with_coords(curr_obs_coords_lst, template_model_map_lst):
    """
        Takes a template model with '{}' and a curr_obs_coords_lst list of model component corrdinates and interps the coords into the modal_map
    """
    assert len(curr_obs_coords_lst) == len(template_model_map_lst), "curr_obs_coords_lst and template_model_map_lst must have equal lengths!"

    new_model_lst = []
    for i, curr_obs_coords in  enumerate(curr_obs_coords_lst):
        # print(f"curr_obs_coords_lst[i] {curr_obs_coords_lst[i]}")
        x, y, z = curr_obs_coords
        # print(f"template_model_map_lst[i] {template_model_map_lst[i] }")
        # template_model_map_lst[i] = template_model_map_lst[i].format(str(x), str(y), str(z))
        new_row = template_model_map_lst[i].format(str(x), str(y), str(z))
        new_model_lst.append(new_row)
        # print(new_row)
        # new_model_lst.append(template_model_map_lst[i].format(str(x), str(y), str(z)))


    return '\n'.join(new_model_lst)

def read_and_return_model_map_lst(path):
    new_model_map_lst = []
    with open(path, "r") as f:
        for model_map in f.readlines():
            if '\n' in model_map:
                new_model_map_lst.append(model_map[:-1])
            else:
                new_model_map_lst.append(model_map)

    return new_model_map_lst


def get_curr_obs_coords_lst_from_model_map_lst(path):
    new_model_map_lst = read_and_return_model_map_lst(path)
    curr_obs_coords_lst = []
    for map_list in new_model_map_lst:
        # print(f"map_list.split(' '): {map_list.split(' ')}")
        x, y, z = map_list.split(' ')[1:4]
        curr_obs_coords_lst.append([int(x), int(y), int(z)])

    return curr_obs_coords_lst


def get_template_from_model_map_lst(path):
    new_model_map_lst = read_and_return_model_map_lst(path)

    # place {} in x, y, z coordinates
    template_model_map_lst = []
    for map_list in new_model_map_lst:
        new_component_row = map_list.split(' ')[0] + " {} {} {} " + ' '.join(map_list.split(' ')[4:])
        template_model_map_lst.append(new_component_row)

    return template_model_map_lst


def write_obs_to_mpd_file(path, curr_obs_coords_lst, template_model_map_lst):
    new_map = interpolate_template_map_with_coords(curr_obs_coords_lst, template_model_map_lst)

    with open(path, "w") as f:
        f.write(new_map)         



