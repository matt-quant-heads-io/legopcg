import random

def predict_action(obs):
    return random.randint(0,2)

obs = [[]]

# making play observation space that is 10 x 10 x 10 x 3
action_block = [0, 0, 1]

x = []
for i in range(10):
    x.append(action_block)

y = []
for i in range(10):
    y.append(x)


z = []
for i in range(10):
    z.append(y)

obs = z
is_solved = False

# agent interacting witht he environment
while True:
    x, y, z, = 0, 0, 0
    action_idx = agent.predict_action(obs) #returns an integer 0,1,2
    action_one_hot = [0]*3
    action_one_hot[action_idx] = 1 #[0,1,0] for example

    #update the environment i.e. step and get the tuple <s, a, s', r>
    obs[z][y][x] = action_one_hot

    # check if we are done? (agent solved or max iterations threshold met)
    if done:
        if solved:
            is_solved = True

        break

    # update our position; Update Observation state?
    x,y,z = update_position(x,y,z)


if is_solved:
    write_dot_dat_file(obs)


     










