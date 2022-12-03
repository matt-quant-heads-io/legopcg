def write_curr_obs_to_dir_path(dir_path, curr_step_num, curr_obs):
    with open(f'{dir_path}/{curr_step_num}', 'w') as f:
       curr_obs_as_char_map = convert_curr_obs_to_char(curr_obs)
       f.write(curr_obs_as_char_map)
