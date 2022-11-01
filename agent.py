import torch
import random

import numpy as np

from game import SnakeGameAI, Direction, Point

from collections import deque # Data Structure to store memory

from model import Linear_QNet, QTrainer

from helper import plot

# Parameters const
# Store 100000 items
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        # Need to store game and model in this class
        
        self.n_games = 0
        self.epsilon = 0 # COntrols the randomness
        self.gamma = 0.9 # Discount rate smaller than 1  
        self.memory = deque(maxlen=MAX_MEMORY) # Remove elements from left when we exceed this mem i.e. popleft() is called

        self.model = Linear_QNet(11, 256, 3) # 11 input, 3 output since 3 different actions, hidden size u can play around with
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # Gets the game
    # We store 11 values
    # Direction only 1 of them is 1
    def get_state(self, game):

        # First item in list is our head
        head = game.snake[0]

        # Create points in all 4 directions around head
        # BLOCK SIZE 20
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Only 1 of them is 1
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # 11 states dependent on current state
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            # Only 1 of them is true
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food is left of us
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        # Convert list to numpy to convert T/F bool to 0/1
        return np.array(state, dtype=int)


    # Function to remember these input params in our memory deque
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # If this exceed curr mem, popleft(). 1 tuple append

    def train_long_memory(self):
        # Grab 1000 samples/tuples from mem sice Batch size 1000

        if len(self.memory) > BATCH_SIZE:
            # Random sample if we already have 1000 elem
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples returned here
        else:
            mini_sample = self.memory # if not 1000 tuples yet
        
        # Extract 
        states, actions, rewards, next_states, dones = zip(*mini_sample) # Can do for loop iteration also
        
        self.trainer.train_step(states, actions, rewards, next_states, dones)



    # 1 game step training
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games

        # Initialize
        final_move = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            
            
            # MOre the games, smaller the epsilon. 
            # This loop is less frequent. So no more random move

            # Random move
            move = random.randint(0,2)
            final_move[move] = 1  #Set this index to 1

        else:
            # Move based on the model: 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # Will execute the forward() which is the prediction

            move = torch.argmax(prediction).item() # Convert tensor to 1 num

            final_move[move] = 1

        return final_move

# Global function
def train():
    
    # Keep track of scores
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 #Best score

    agent = Agent()

    game = SnakeGameAI()

    # Training Loop
    while True:

        # Get current/old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move
        reward, done, score = game.play_step(final_move)
        # New state from new game 
        state_new = agent.get_state(game)

        # Train short memory - only for 1 step
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember in deque
        agent.remember(state_old, final_move, reward, state_new, done)
        
        # If game over
        if done:
            # Train the long memory (Replay-memory; Experienced Replay)
            # Trains again on all prev moves and games it played/
            # Helps to improve itself
            game.reset()
            agent.n_games += 1

            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', 'Record:', record)

            # Append Score after each game
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()