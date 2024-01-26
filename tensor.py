import numpy as np
import gym
import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up game constants
width, height = 600, 400
snake_size = 20
fps = 10

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)

# Create the game window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

clock = pygame.time.Clock()


class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((width // 2), (height // 2))]
        self.direction = (1, 0) 

    def get_head_position(self):
        return self.positions[0]

    def update(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + (x * snake_size)) % width), (cur[1] + (y * snake_size)) % height)
        if len(self.positions) > 2 and new in self.positions[2:]:
            self.reset()
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()

    def reset(self):
        self.length = 1
        self.positions = [((width // 2), (height // 2))]
        self.direction = pygame.KEYDOWN

    def render(self, surface):
        for p in self.positions:
            pygame.draw.rect(surface, white, (p[0], p[1], snake_size, snake_size))


class SnakeGame(gym.Env):
    def __init__(self):
        super(SnakeGame, self).__init__()
        self.snake = Snake()
        self.food = self.generate_food()
        self.state_space = width // snake_size, height // snake_size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_space[0], self.state_space[1], 3),
                                                dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)

    def step(self, action):
        self.snake.direction = action
        self.snake.update()

        done = False
        reward = 0

        if self.snake.get_head_position() == self.food:
            self.food = self.generate_food()
            self.snake.length += 1
            reward = 1

        if len(self.snake.positions) > 1 and self.snake.get_head_position() in self.snake.positions[1:]:
            done = True
            reward = -1

        if not (0 <= self.snake.get_head_position()[0] < width and 0 <= self.snake.get_head_position()[1] < height):
            done = True
            reward = -1

        state = self.get_state()
        return state, reward, done, {}

    def reset(self):
        self.snake.reset()
        self.food = self.generate_food()
        return self.get_state()

    def render(self, mode='human'):
        screen.fill(black)
        self.snake.render(screen)
        pygame.draw.rect(screen, white, (*self.food, snake_size, snake_size))
        pygame.display.flip()
        clock.tick(fps)

    def generate_food(self):
        food_position = (np.random.randint(0, width // snake_size) * snake_size,
                         np.random.randint(0, height // snake_size) * snake_size)
        while food_position in self.snake.positions:
            food_position = (np.random.randint(0, width // snake_size) * snake_size,
                             np.random.randint(0, height // snake_size) * snake_size)
        return food_position

    def get_state(self):
        state = np.zeros((self.state_space[0], self.state_space[1], 3), dtype=np.uint8)
        for p in self.snake.positions:
            state[p[0] // snake_size, p[1] // snake_size, 0] = 1
        state[self.food[0] // snake_size, self.food[1] // snake_size, 1] = 1
        state[self.snake.get_head_position()[0] // snake_size, self.snake.get_head_position()[1] // snake_size, 2] = 1
        return state


# Create Snake Game Environment
env = SnakeGame()

# Q-network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_shape=(width // snake_size, height // snake_size, 3)))
model.add(Dense(128, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

model.compile(optimizer='adam', loss='mse')

# Training the Q-network
import random
from collections import deque

memory = deque(maxlen=1000)
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95
batch_size = 32

for episode in range(1000):
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    total_reward = 0

    for time_step in range(1000):  # You can adjust the maximum number of time steps per episode
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])

        next_state, reward, done, _ = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        total_reward += reward

        memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            break

        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + gamma * np.amax(model.predict(next_state)[0]))
                target_f = model.predict(state)
                target_f[0][action] = target
                model.fit(state, target_f, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Test the trained Q-network
state = env.reset()
state = np.expand_dims(state, axis=0)

while True:
    action = np.argmax(model.predict(state)[0])
    next_state, _, done, _ = env.step(action)
    next_state = np.expand_dims(next_state, axis=0)
    state = next_state

    env.render()

    if done:
        pygame.quit()
        sys.exit()
