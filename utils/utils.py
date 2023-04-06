import numpy as np
import os
import subprocess
import shutil
import imageio
import re
import torch 



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

def convert_one_hot_to_char(one_hot_map):
    char_map = []

    for y in range(len(one_hot_map)):
        for x in range(len(one_hot_map[0])):
            for z in range(len(one_hot_map[0][0])):
                char_tile = onehot_index_to_char_map[one_hot_map[y][x][z]]
                char_map.append(char_tile)

    return np.array(char_map).reshape((10,10,10))
    
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
               
def cleanup_dir(dir_path):
    try: 
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except FileNotFoundError:
        print("Directory does not exist, skipping cleanup")


def write_curr_obs_to_dir_path(obs_one_hot, dir_path, curr_step_num, lego_block_dims):
    # cleanup_dir(dir_path)
    char_map = convert_one_hot_to_char(obs_one_hot)
    f = open(dir_path+'/step_'+str(curr_step_num)+'.txt', "a")
    f.write(str(char_map))
    f.close()
    generate_dat_file(char_map, dir_path,curr_step_num,lego_block_dims)

def generate_dat_file(char_map, dir_path,curr_step_num, lego_block_dims):
    f = open(dir_path+'/step_'+str(curr_step_num)+'.ldr', "a")
    f.write("0\n")
    f.write("0 Name: New Model.ldr\n")
    f.write("0 Author:\n")
    f.write("\n")

    
    start_block_char = char_map[0][0][0]
    start_block_name = char_to_str_map[start_block_char]
    start_block_dimensions = lego_block_dims[start_block_name]
    
    for y in range(len(char_map[0])):
        for x in range(len(char_map)):
            for z in range(len(char_map[0][0])):
                char = char_map[y][x][z]

                if (char != 'w'):
                    # Get Dimensions of Lego Block
                    lego_block_name = char_to_str_map[char]
                    current_xyz_dims = lego_block_dims[lego_block_name]

                    # Along x-dirn
                    factor = 0
                    if (start_block_dimensions[0] != 0):
                        if (current_xyz_dims[0] > start_block_dimensions[0]):
                            factor = (current_xyz_dims[0] - start_block_dimensions[0]) * 10
                        elif (current_xyz_dims[0] < start_block_dimensions[0] ): # was >0
                            factor = (current_xyz_dims[0] - start_block_dimensions[0]) * 10  
                            

                    
                        
                    x_lego = x * 20 + factor
                    y_lego = y * -24  
                    # y_lego = y * 24 * currentXYZDims[1] 
                    z_lego = z * 20 

                    

                if (char == 'w'):
                    a=1

                elif (char == 'a'):
                    f.write("1 7 ")
                    
                    f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
                    f.write("1 0 0 0 1 0 0 0 1 ")
                    f.write(getBlockName(char) + ".dat")
                    f.write("\n")

                elif (char == 'b'):
                    f.write("1 7 ")
                    f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
                    f.write("1 0 0 0 1 0 0 0 1 ")
                    f.write(getBlockName(char) + ".dat")
                    f.write("\n")

                elif (char == 'c'):
                    f.write("1 7 ")
                    f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
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

def create_dir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Directory already exists, skipping creation") 


# def render_in_blender(lego_block_coords : list[list], call_counter : int):
#     """
#         Render the scene in blender
#     """

#     # print(os.getcwd())
#     script_path = f"./animations/render_script_{call_counter:02d}.py"
#     images_path = f"./animations/images{call_counter:02d}/"
#     create_dir(images_path)

#     with open(script_path, "w") as outfile:
        
#         script = f"""
# import bpy
# # # Set up the scene
# # bpy.ops.object.delete(use_global=False, confirm=False)
# # scene = bpy.context.scene
# # scene.camera.location = (0, 0, 15)

# image_counter = 0 
# image_path = "{images_path}"
# bpy.ops.object.select_all(action='DESELECT')
# # Select the default cube
# bpy.data.objects['Cube'].select_set(True)
# # Delete the default cube
# bpy.ops.object.delete()

# # Set up the camera
# camera = bpy.data.objects['Camera']
# camera.location = (15, -15, 15)
# camera.rotation_euler = (0.7854, 0, 0.7854)

# # set render resolution
# bpy.context.scene.render.resolution_x = 500
# bpy.context.scene.render.resolution_y = 500
#         """
#         outfile.write(script)
#         for y,x,z in lego_block_coords:
#             script = f"""
# bpy.ops.mesh.primitive_cube_add(size=1, location=({x}, {z}, {y}))
# path = image_path + "image_" + "{{:02d}}".format(image_counter) + ".png"
# bpy.context.scene.render.filepath = path
# bpy.ops.render.render(write_still=True, use_viewport=True)
# image_counter += 1
# """
#             outfile.write(script)
    
#     blender_app = "/Applications/Blender.app/Contents/MacOS/Blender"
#     args = [blender_app, "--background", "--python", script_path]
    
#     result = subprocess.run(args=args, check=True)

#     if result.returncode == 0:
#         create_gif(images_path, call_counter)
#     else:
#         print("Error in rendering the scene in blender")



# def render_in_blender(lego_block_coords, call_counter : int):
#     """
#         Render the scene in blender
#     """

#     # print(os.getcwd())
#     script_path = f"./animations/render_script_{call_counter:02d}.py"
#     images_path = f"./animations/images{call_counter:02d}/"
#     create_dir(images_path)

#     with open(script_path, "w") as outfile:
        
#         script = f"""
# import bpy
# import math
# import mathutils

# def setupCamera(scene, c):
#     pi = math.pi

#     scene.camera.rotation_euler[0] = c[0] * (pi / 180.0)
#     scene.camera.rotation_euler[1] = c[1] * (pi / 180.0)
#     scene.camera.rotation_euler[2] = c[2] * (pi / 180.0)

#     scene.camera.location.x = c[3]
#     scene.camera.location.y = c[4]
#     scene.camera.location.z = c[5]

#     return

# image_counter = 0 
# image_path = "{images_path}"
# bpy.ops.object.select_all(action='DESELECT')
# # Select the default cube
# bpy.data.objects['Cube'].select_set(True)
# # Delete the default cube
# bpy.ops.object.delete()

# # Set up the camera
# # camera = bpy.data.objects['Camera']

# scene = bpy.data.scenes["Scene"]

# config = list([67.1349, 0.779594, 148.858, 15.57961, 19.16202, 25.34536])

# bpy.ops.object.camera_add()
# cam = bpy.data.objects['Camera']
# cam.rotation_mode = 'XYZ'

# scene.camera = cam

# setupCamera(scene=scene, c=config)



# # set render resolution
# bpy.context.scene.render.resolution_x = 500
# bpy.context.scene.render.resolution_y = 500
#         """
#         outfile.write(script)
#         for y,x,z in lego_block_coords:
#             script = f"""
# # revolveCameraByFractionOfCircle(camera, 1.0, ({x}, {z}, {y}))
# bpy.ops.mesh.primitive_cube_add(size=1, location=({x}, {z}, {y}))
# path = image_path + "image_" + "{{:02d}}".format(image_counter) + ".png"
# bpy.context.scene.render.filepath = path
# bpy.ops.render.render(write_still=True, use_viewport=True)
# image_counter += 1
# """
#             outfile.write(script)
    
#     blender_app = "/Applications/Blender.app/Contents/MacOS/Blender"
#     args = [blender_app, "--background", "--python", script_path]
    
#     result = subprocess.run(args=args, check=True)

#     if result.returncode == 0:
#         create_gif(images_path, call_counter)
#     else:
#         print("Error in rendering the scene in blender")
    
def render_in_leocad(base_path, env_num, lego_block_details):
    """
        Renders the lego block coords in leocad
    """

    leocad_path = os.path.join(base_path, f"leocad_{env_num:02d}")
    image_path = os.path.join(base_path, f"images_{env_num:02d}")

    block_counter = len(lego_block_details)

    if block_counter == 1:
        create_dir(leocad_path)
        create_dir(image_path)

    # counter = 0 
    factor = 0

    dat_file = os.path.join(leocad_path, f'block_{block_counter:02d}.dat')
    png_file = os.path.join(image_path, f'block_{block_counter:02d}.png')
                             
    with open(dat_file, 'w') as f:
        for y, x, z, brick_type in lego_block_details:
            lego_block_name = onehot_index_to_str_map[brick_type]
            # counter += 1 
            x_lego = x * 20 + factor
            y_lego = y * -24  
            z_lego = z * 20 

            line1 = ("1 7 ")
            line2 = str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' '
            line3 = "1 0 0 0 1 0 0 0 1 "
            line4 = lego_block_name+ ".dat\n"
            f.write(line1 + line2 + line3 + line4)

    def generate_png():
        # leocad app path in mac - convert it to a shell command
        app_path = "/Applications/LeoCAD.app/Contents/MacOS/LeoCAD"
        # args = [app_path, "--background", "--python", script_path]
        leocad_command = f'''
            {app_path}	\
            --height "2046"	\
            --width "2046"	\
            --camera-angles 30 0	\
            --shading "full" \
            --line-width "2" \
            --aa-samples "8" \
            --image "{png_file}" \
            {dat_file}																	

        '''
        os.system(leocad_command)
    
    generate_png()

def create_gif(base_path):
    """
        look for image folders in the base path and create a gif 
        for each folder
    """

    for _, folders, _ in os.walk(base_path, topdown=True):
        for folder in folders:
            if folder.startswith("images"):
                files = []
                images = []
                images_path = os.path.join(base_path, folder)
                for file_name in os.listdir(images_path):
                    if file_name.endswith(".png"):
                        files.append(file_name)
                
                # sort the files in the order of image number
                files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

                for file_name in files:
                        file_path = os.path.join(images_path, file_name)
                        images.append(imageio.v2.imread(file_path))
                
                imageio.mimsave(os.path.join(images_path, f"animation.gif"), 
                                images, fps=2)