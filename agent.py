import numpy as np
import pygame
import sys
from tensorflow.keras.models import load_model
import game
import model
import random
# Initialize Pygame
pygame.init()

# Set up game constants
width, height = 600, 400
snake_size = 20
fps = 10

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)
history = ""
# Load the trained Q-network
# model = model.make_model((4,), 1, 3)


optimizer = "adam"
# Load the trained Q-network
q_network = model.make_model((6,), 1, 4)
q_network.compile(optimizer=optimizer, loss="mse")
# Initialize other parameters
gamma = 0.9  # Discount factor for future rewards
  # You can adjust the optimizer and learning rate as needed
learning_rate = 0.001
batch_size = 32
replay_memory = []  # Buffer to store experiences for online training

env = game.Game()

# Function to add an experience to replay memory
def add_experience(state, action, reward, next_state, done):

    replay_memory.append([state, action, reward, next_state, done])


    # print(f"mem:{replay_memory}")
    # Ensure replay memory does not exceed specified capacity
    if len(replay_memory) > 10000:
        replay_memory.pop(0)
# Online training function
def train_model():
    global history
    if len(replay_memory) < batch_size:
        return  # Not enough experiences for training yet

    batch_indices = np.random.choice(len(replay_memory), batch_size, replace=False)
    batch = [replay_memory[i] for i in batch_indices]



    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert the arrays to NumPy arrays
    states = np.vstack(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.vstack(next_states)
    dones = np.array(dones)

    # Calculate target Q-values using the Bellman equation
    target_values = rewards + gamma * np.amax(q_network.predict(next_states), axis=1) * (1 - dones)

    # Get the current Q-values for the chosen actions
    current_values = q_network.predict(states)
    chosen_action_values = current_values[np.arange(batch_size), actions]

    # Update the Q-values using the temporal difference error
    current_values[np.arange(batch_size), actions] = target_values

    # Train the model on the updated Q-values
    q_network.train_on_batch(states, current_values)
    history = q_network.fit(states, current_values, verbose=0)

    # Print the training metrics
    # print("Training Metrics:", history.history)
# Test the trained Q-network
state = env.reset()



env = game.Game()

# Test the trained Q-network
state = env.get_state()
tries = 0
while True:
    action = np.argmax(q_network.predict(np.expand_dims(state, axis=0))[0])
    next_state, reward, done = env.step(action)
    add_experience(state, action, reward, next_state, done)
    train_model()  #
    state = next_state
    if history:
        env.set_caption(f"Tries:{tries}  Loss:{history.history}")

    env.draw()
    if done == 1:
        tries += 1

        env.reset()
