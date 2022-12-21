import map_encodings
import os
import sys
import subprocess
import imageio

import re

import LegoDimensions


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
    charMap = map_encodings.convertOneHotToChar(obsOneHot)
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
    startBlockName = map_encodings.char_to_str_map[startBlockChar]
    startBlockDimensions = LegoDimensions.LegoDimsDict[startBlockName]
    
    for y in range(len(charMap[0])):
        for z in range(len(charMap)):
            for x in range(len(charMap[0][0])):
                char = charMap[z][y][x]

                if (char != 'w'):
                    # Get Dimensions of Lego Block
                    legoBlockName = map_encodings.char_to_str_map[char]
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
                    f.write(map_encodings.getBlockName(char) + ".dat")
                    f.write("\n")

                elif (char == 'b'):
                    f.write("1 7 ")
                    f.write(str(xLego) + ' ' + str(yLego) + ' ' + str(zLego) + ' ')
                    f.write("1 0 0 0 1 0 0 0 1 ")
                    f.write(map_encodings.getBlockName(char) + ".dat")
                    f.write("\n")

                elif (char == 'c'):
                    f.write("1 7 ")
                    f.write(str(xLego) + ' ' + str(yLego) + ' ' + str(zLego) + ' ')
                    f.write("1 0 0 0 1 0 0 0 1 ")
                    f.write(map_encodings.getBlockName(char) + ".dat")
                    f.write("\n")
                
    f.close()            



