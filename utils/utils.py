#from typing import Self
import numpy as np
import os
import sys
import subprocess
import imageio
import re
import torch 
import matplotlib.pyplot as plt
import imageio



from scipy.interpolate import make_interp_spline



LegoDimsDict = {
    "empty" : [0,0,0],
    "3005" : [1,3,1], # x,y,z dims
    "3004" : [2,3,1],
    "3003" : [2,3,2],
    "3031" : [4,1,4], #1 y coordinate unit is 8 LDU
}


onehot_index_to_onehot_map = {
    0 : [0,0,0,0],    
    1 : [0,0,0,1],
    2: [0,0,1,0],
    3: [0,1,0,0],
    4: [1,0,0,0],
}

#def getBlockName(char):
#    return char_to_str_map[char]

str_to_onehot_index_map = { 
    'empty': 0,    
    '3005': 1,
    '3004': 2,
    '3003': 3,
    '3031': 4,
}

onehot_index_to_str_map = { 
    0: 'empty',    
    1: '3005',
    2: '3004',
    3: '3003',
    4: '3031',
}

def get_device() -> str:
    """
        Return device name based on the availability of GPU
    """
    # support for cuda and apple silicon
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.has_mps:
        # device = torch.device('mps')
        # torch has not yet completely implemented for mps 
        device = torch.device('cpu')

    else:
        device = torch.device('cpu')

    # print(device)
    return device


def renderpng(mpdfile, imgname):
    ##angle was 30 30
    leocad_command = f'''
	    leocad	\
		--height "2046"	\
		--width "2046"	\
		--camera-angles 30 30	\
		--shading "full" \
		--line-width "2" \
		--aa-samples "8" \
		--image "{imgname}" \
		{mpdfile}																	

    '''
    os.system(leocad_command)

def save_arrangement(blocks, dir_path, curr_step_num, curr_reward, rewards = None, render = False, episode = None, goal = None):

    if not os.path.exists(dir_path + "mpds"):
            os.makedirs(dir_path + "mpds")

    if not os.path.exists(dir_path + "images"):
            os.makedirs(dir_path + "images")

    savedir = dir_path + "mpds/"
    imgdir = dir_path + "images/"

    if rewards:
        x = range(len(rewards))
        plt.title("reward")
        plt.plot(x, rewards)
        plt.savefig(dir_path + "/rewards.png")
        

        
    filename = f"{curr_step_num:05}"#str(curr_step_num) 


    if episode != None:
        filename = f"{episode:04}_" + filename
    if curr_reward != None:
        filename += "_"+ str(curr_reward) 
    if goal != None:
        filename += "_x" + str(goal[0])+"z"+str(goal[1])
    f = open(savedir + filename +'.mpd', "a")
    f.write("0\n")
    f.write("0 Name: New Model.ldr\n")
    f.write("0 Author:\n")
    f.write("\n")
    for x in range(15):
        for z in range(15):
            lego_block_name = "3005"
            current_xyz_dims = [1,3,1]
            block_color = "2 "
            
            y_offset = -3#-24

            x_lego = x * 20  + 10*(LegoDimsDict[lego_block_name][0]-1) #- 5*10
            y_lego =  0#(1)*(LegoDimsDict[lego_block_name][1])
            z_lego = z * 20 + 10*(LegoDimsDict[lego_block_name][2]-1) #- 5*10

            #print(block.x, block.y, block.z)
            
            f.write("1 ")
            f.write(block_color)
            f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
            f.write("1 0 0 0 1 0 0 0 1 ")
            f.write(lego_block_name + ".dat")
            f.write("\n")


    for block in blocks:
        lego_block_name = block.block_name
        current_xyz_dims = block.dims
        block_color = "7 "
        if block.is_next_block :
            block_color = "14 "
        elif block.is_curr_block: #solid yellow
            if block.error == "stay":
                block_color = "25 " #orange
            elif block.error == "bounds":
                block_color = "69 " #purple
            elif block.error == "overlap":
                block_color = "1 " #blue
            else:
                block_color = "46 " #transparent yellow

        y_offset = -24

        x_lego = block.x * 20  + 10*(LegoDimsDict[lego_block_name][0]-1) #- 5*10
        y_lego = block.y * (-24)/3  + (y_offset/3)*(LegoDimsDict[lego_block_name][1])
        z_lego = block.z * 20 + 10*(LegoDimsDict[lego_block_name][2]-1) #- 5*10

        #print(block.x, block.y, block.z)
        
        f.write("1 ")
        f.write(block_color)
        f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
        f.write("1 0 0 0 1 0 0 0 1 ")
        f.write(lego_block_name + ".dat")
        f.write("\n")

    f.close() 

    if render: 
        renderpng(savedir + filename + ".mpd", imgdir + filename + ".png")

    


def animate(dir, iter = None):
    print("animating " + str(iter))
    images = []

    savedir = dir
    if iter != None:
        savedir = savedir + str(iter) + "/"
    if not os.path.exists(savedir):
            print("making animation directory for " + str(iter))
            os.makedirs(savedir)

    filenames = sorted([f for f in os.listdir(savedir + "images") if f.split(".")[-1] == "png"])
    images = []
    for filename in filenames:
        images.append(imageio.imread(savedir + "images/" + filename))
    
    savename = "animation.gif"
    if iter != None:
        savename = str(iter) + "_" + savename

    try:
        imageio.mimsave(dir +  savename, images, fps = 5)
    except:
        return