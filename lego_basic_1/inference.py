# load in trained model

# define environment instance
    # random starting obs (state) 
NUM_TRIALS = 10000
NUM_SUCCESS = NUM_FAILS = 0
GENERATED_MAPS_PATH = 'playable_maps'



agent = torch.load('path_to_model')
env = LegoPCGEnv()
obs = env.reset()

for trial in range(NUM_TRIALS):
    while True:
        action = agent.predict(curr_obs)
        obs, _, is_finished, info = env.step(action) #TODO: impl info['solved']

        if is_finished:
            if info['solved']:
                NUM_SUCCESS += 1
                str_map = get_string_map(curr_obs) #returns ["wall", ...]
                char_map = get_char_map(str_map) # see char_map_example.txt
                save_map(char_map, path)
            else:
                NUM_FAILS += 1
            curr_obs = env.reset()
            break
            


print(f"Percentage playable levels: {NUM_SUCCESS/NUM_TRIALS}")


