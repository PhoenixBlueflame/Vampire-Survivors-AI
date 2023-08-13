import gym
from gym import spaces
import pyautogui
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import optimizers, losses
from collections import deque
import random
import time
import os
import pickle

QUICK_START_POSITION = (1867, 1029)
REVIVE_BUTTON_POSITION = (1280, 1400)
QUIT_BUTTON_POSITION = (1300, 1100)
DONE_BUTTON_POSITION = (1440, 1500)
REROLL_BUTTON_POSITION = (2150, 500)
LEVELUP_BUTTON_POSITION = (1400, 500)
TOPEN_BUTTON_POSITION = (1270, 1300)
TDONE_BUTTON_POSITION = (1270, 1300)
GAMMA = 0.99
HP_BAR_COORDS = (1385, 825, 114, 1)
current_actions = []

class VampireSurvivorsEnv(gym.Env):
    def __init__(self):
        super(VampireSurvivorsEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(8)  # 8 actions: Up, Down, Left, Right, NE, SE, NW, SW
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def reset(self):
        # Return the initial observation
        return get_screen()

    def step(self, action):
        # Execute the given action in the game
        execute_action(action)

        # Capture the next state (game screen)
        next_state = get_screen()

        # Initialize reward and done status
        reward = 0
        done = False

        # NOTE: We don't check game state here as it's done outside in the main loop
        reward = 0.1  # Default reward when the agent is just playing without any event

        return next_state, reward, done, {}

    def render(self, mode='human'):
        # For this setup, the game is already being rendered so you might not need to do anything here
        pass

    def close(self):
        # Close or minimize the game (placeholder logic)
        pass

def save_model(model, filename):
    model.save_weights(filename)

def load_model(model, filename):
    model.load_weights(filename)

def save_replay_memory(memory, filename):
    with open(filename, 'wb') as f:
        pickle.dump(memory.memory, f)

def load_replay_memory(memory, filename):
    with open(filename, 'rb') as f:
        memory.memory = pickle.load(f)

def train_dqn_batch(model, target_model, experiences):
    # Extracting experience from the input
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = experiences

    # Compute the current Q-values
    current_q_values = model.predict(state_batch)

    # Compute the next Q-values using the target network
    next_q_values = target_model.predict(next_state_batch)

    # Compute the target Q-values
    target_q_values = current_q_values.copy()

    for i in range(len(state_batch)):
        # For terminal states, the Q-value is just the reward
        if done_batch[i]:
            target_q_values[i, action_batch[i]] = reward_batch[i]
        # For non-terminal states, the Q-value is the immediate reward plus the discounted future reward
        else:
            target_q_values[i, action_batch[i]] = reward_batch[i] + GAMMA * np.max(next_q_values[i])

    # Train the model
    model.train_on_batch(state_batch, target_q_values)

def get_screen():
    screenshot = pyautogui.screenshot()
    game_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    game_image_gray = cv2.cvtColor(game_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(game_image_gray, (84, 84))

    return resized_image


def execute_action(action):
    global current_actions

    actions = ["up", "down", "left", "right", "ne", "se", "nw", "sw"]
    selected_action = actions[action]

    # Release previously pressed keys
    for act in current_actions:
        if act in ["up", "down", "left", "right"]:
            pyautogui.keyUp(act)

    # Reset the current_actions list
    current_actions = []

    print(f"Taken action is: {selected_action}")
    # Check if it's a diagonal move
    if selected_action == "ne":
        pyautogui.keyDown("up")
        pyautogui.keyDown("right")
        current_actions.extend(["up", "right"])
    elif selected_action == "se":
        pyautogui.keyDown("down")
        pyautogui.keyDown("right")
        current_actions.extend(["down", "right"])
    elif selected_action == "nw":
        pyautogui.keyDown("up")
        pyautogui.keyDown("left")
        current_actions.extend(["up", "left"])
    elif selected_action == "sw":
        pyautogui.keyDown("down")
        pyautogui.keyDown("left")
        current_actions.extend(["down", "left"])
    else:
        pyautogui.keyDown(selected_action)
        current_actions.append(selected_action)


def build_dqn_model(input_shape, n_actions):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.memory)

def epsilon_greedy_policy(model, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        q_values = model.predict(state[np.newaxis])
        return np.argmax(q_values[0])

def is_color_close_to_green(rgb):
    r, g, b = rgb
    return g > 180 and r < 63 and b < 92

def is_color_close_to_red(rgb):
    r, g, b = rgb
    return g < 44 and r > 210 and b < 13

def is_color_close_to_blue(rgb):
    r, g, b = rgb
    return g < 65 and r < 40 and b > 204

def calc_hp_percentage():
    # Capture the region defined by HP_BAR_COORDS
    hp_bar_image = pyautogui.screenshot(region=HP_BAR_COORDS)
    #hp_bar_image.save("hp_bar_screenshot.png")

    # Convert the image into an RGB array
    hp_array = np.array(hp_bar_image)

    # Extract the red channel
    red_channel = hp_array[:, :, 0]

    # Create a mask to identify the red pixels in the HP bar
    mask = np.where((red_channel > 223) & (red_channel < 225), 1, 0)

    # Sum the number of red pixels along the width
    red_pixels_width = np.sum(mask)

    # Calculate HP ratio
    hp_ratio = red_pixels_width / HP_BAR_COORDS[2]  # Assuming HP_BAR_COORDS[2] is the width

    return hp_ratio

def calc_state():
    reroll_pixel_color = pyautogui.pixel(REROLL_BUTTON_POSITION[0], REROLL_BUTTON_POSITION[1])

    if is_color_close_to_blue(reroll_pixel_color):
        pyautogui.click(LEVELUP_BUTTON_POSITION)
        return 'level-up'

    revive_pixel_color = pyautogui.pixel(REVIVE_BUTTON_POSITION[0], REVIVE_BUTTON_POSITION[1])

    if is_color_close_to_green(revive_pixel_color):
        pyautogui.click(REVIVE_BUTTON_POSITION)
        return 'revive'

    quit_pixel_color = pyautogui.pixel(QUIT_BUTTON_POSITION[0], QUIT_BUTTON_POSITION[1])

    if is_color_close_to_red(quit_pixel_color):
        pyautogui.click(QUIT_BUTTON_POSITION)
        time.sleep(0.5)
        pyautogui.click(DONE_BUTTON_POSITION)
        time.sleep(2)
        pyautogui.press('space')
        return 'died'

    treasure_pixel_color = pyautogui.pixel(TOPEN_BUTTON_POSITION[0], TOPEN_BUTTON_POSITION[1])

    if is_color_close_to_blue(treasure_pixel_color):
        pyautogui.click(TOPEN_BUTTON_POSITION)
        time.sleep(18)
        pyautogui.click(TDONE_BUTTON_POSITION)
        return 'treasure'

    return 'alive'

def is_on_main_menu():
    return True

def main():
    # DQN logic parameters
    n_actions = 8
    input_shape = (84, 84, 1)
    batch_size = 32
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.1
    gamma = 0.99
    training_frequency = 4
    target_update_frequency = 1000
    replay_memory_capacity = 10000
    num_episodes = 10000
    optimizer = optimizers.Adam(learning_rate=0.001)
    loss_fn = losses.MeanSquaredError()

    model = build_dqn_model(input_shape, n_actions)
    target_model = build_dqn_model(input_shape, n_actions)
    target_model.set_weights(model.get_weights())
    replay_memory = ReplayMemory(replay_memory_capacity)

    env = VampireSurvivorsEnv()

    overall_step = 0

    load_model_path = "dqn_model_weights.h5"
    load_memory_path = "replay_memory.pkl"

    if os.path.exists(load_model_path):
        load_model(model, load_model_path)
        print("Loaded pre-trained weights.")

    if os.path.exists(load_memory_path):
        load_replay_memory(replay_memory, load_memory_path)
        print("Loaded saved replay memory.")

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        levelups = 0
        prev_hp = 1

        # Initialize the time checkpoint at the start of the episode
        last_check_time = time.time()

        if episode % 2 == 0:  # save every 100 episodes
            save_model(model, f"dqn_model_weights_{episode}.h5")
            print(f"Saved model weights at Episode {episode}")

            save_replay_memory(replay_memory, f"replay_memory_{episode}")
            print(f"Saved replay memory at Episode {episode}")

            pyautogui.click(748, 507)
            pyautogui.press('f11')
            time.sleep(0.3)
            pyautogui.press('f11')

            time.sleep(2)

        pyautogui.click(QUICK_START_POSITION)



        while not done:
            action = epsilon_greedy_policy(model, state, epsilon, n_actions)
            next_state, _, done, _ = env.step(action)
            reward = 0

            # Check the time since the last checkpoint
            current_time = time.time()

            if current_time - last_check_time >= 1:  # 5 seconds
                game_state = calc_state()
                if game_state == 'died':
                    reward = -100
                    state = env.reset()  # Reset the environment if character died
                    done = True
                elif game_state == 'revive':
                    reward += -50
                elif game_state == 'level-up':
                    levelups += 1
                    reward += 5 * levelups
                elif game_state == 'treasure':
                    reward += 100
                else:
                    reward = 1
                    hp_percentage = calc_hp_percentage()
                    print(f"Current HP Percentage: {hp_percentage * 100}%")
                    if hp_percentage < prev_hp:
                        reward += -10

                last_check_time = current_time

            episode_reward += reward

            replay_memory.push(state, action, reward, next_state, done)
            state = next_state

            # Train the model with replay
            if len(replay_memory) >= batch_size and overall_step % training_frequency == 0:
                experiences = replay_memory.sample(batch_size)
                train_dqn_batch(model, target_model, experiences)

            # Update target model occasionally
            if overall_step % target_update_frequency == 0:
                target_model.set_weights(model.get_weights())

            overall_step += 1
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward}")
    env.close()

if __name__ == "__main__":
    main()
