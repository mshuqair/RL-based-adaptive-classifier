from keras.models import Sequential
from keras.layers import Dense, LSTM
from utilities.stream_buffers import StreamBuffer
from utilities.data_stream import DataStream
import pandas as pd
import numpy as np
import random
import time


def build_agent():
    print("Building the agent...\n")
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(state_update_interval, 6)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=2, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


def get_slope(y1, y2):
    if state_update_counter == 1:
        slope = 0
    else:
        slope = (y2 - y1) / (state_update_interval - 1)
    if slope == 0:
        slope = 0.001
    slope = round(slope, 4)
    return slope


def get_state(temp_state):
    periodic_slope = get_slope(temp_state[0, 0], buffer.win_outliers_percent)
    clf_c1_slope = get_slope(temp_state[0, 2], buffer.win_outliers_clf_c1_percent)
    clf_c2_slope = get_slope(temp_state[0, 4], buffer.win_outliers_clf_c2_percent)
    current_state = np.array([buffer.win_outliers_percent, periodic_slope,
                              buffer.win_outliers_clf_c1_percent, clf_c1_slope,
                              buffer.win_outliers_clf_c2_percent, clf_c2_slope])
    temp_state[int(state_update_counter - 1), :] = current_state
    return temp_state


def take_action(act):
    actions = np.array([[False], [True]])  # list of actions for the agent to select from
    new_action = actions[act][0]  # just to convert the array to a scalar value
    return new_action


def get_reward(transition_detected, transition_position, current_position):
    step_reward = None
    true_positive = 5
    true_negative = 1
    false_positive = -1
    false_negative = -5
    reward_cutoff = rewarding_window  # the window size for rewarding when class transition is detected

    # calculate class transition distance for rewarding taking into account multiple transitions
    transition_distance = current_position - transition_position
    if np.all(transition_distance < 0):
        transition_distance = -1
    elif np.all(transition_distance >= 0):
        transition_distance = np.min(transition_distance)
    elif np.any(transition_distance < 0):
        transition_distance = np.max(transition_distance)

    # giving a reward based on whether the detection was within the rewarding window
    if transition_detected is False and transition_distance in range(0, reward_cutoff):
        step_reward = false_negative
    elif transition_detected is False and transition_distance not in range(0, reward_cutoff):
        step_reward = true_negative
    elif transition_detected is True and transition_distance in range(0, reward_cutoff):
        step_reward = true_positive
    elif transition_detected is True and transition_distance not in range(0, reward_cutoff):
        step_reward = false_positive
    step_reward = np.round(step_reward, 4)
    return step_reward


def load_data(file_name):
    data = pd.read_csv(f'data/{file_name}.csv')
    transitions = find_true_transition(data)
    return data, transitions


def find_true_transition(data):
    actual_change = np.array([], dtype='int')
    for j in range(len(data) - 1):
        if data.iloc[j + 1, -1] - data.iloc[j, -1] != 0:  # last column selected for class labels
            actual_change = np.append(actual_change, j + 1)
    return actual_change


# Main Code

# Load data
file_name = 'mhealth_s5'  # name used for saving model, plots, reward, etc...
df, true_transitions = load_data(file_name)
print("Actual label changes at:", true_transitions)

# General parameters
save_model = False  # to save the trained mode
save_reward = False  # to save the cumulative reward
start_negative = True  # when the starting class is 1

# Data window parameters
size = 40  # sliding window size (X_w)
slide = 10  # sliding window step size

# Anomaly detector parameters
nu = 0.2  # nu value for the anomaly detector
kernel = 'rbf'  # kernel function
degree = 3  # degree of the poly function

# RL model parameters
state_update_interval = 4  # steps for the state to be updated before feeding it to agent (u)
rewarding_window = 20  # rewarding window size (t_r)
episodes = 1000  # number of episodes for training
epsilon = 1.0  # epsilon value for the epsilon greedy algorithm
epsilon_min = 0.01  # the minimum epsilon value to reach
epsilon_decay = 1 / episodes  # decreasing epsilon value at each episode
q_gamma = 0.5  # gamma value for the q-learning equation
agent = build_agent()  # building the agent

# General Anomaly detector parameters
clf_general_update_interval = 4

# Initialize parameters
state = np.zeros((state_update_interval, 6))
action = False
clf_periodic_counter = 1
state_update_counter = 1
cumulative_reward = np.zeros(episodes)
epsilon_value = np.zeros(episodes)
episode_number = np.zeros(episodes)

start = time.time()

for episode in range(episodes):
    print(f"Episode number: {episode + 1}")
    print(f"Epsilon value: {epsilon}")
    episode_number[episode] = episode + 1
    epsilon_value[episode] = epsilon

    stream = DataStream(df)
    buffer = StreamBuffer(size, stream.n_features, slide, nu, kernel, degree, start_negative)

    i = 0
    while stream.has_more_samples():
        if buffer.is_full():
            if buffer.at_start():
                buffer.clf_train_general()
                buffer.clf_train_specific()
            else:
                buffer.score_check()
                state = get_state(state)

                if state_update_counter == state_update_interval:
                    state_update_counter = 1
                    q_value = agent.predict(
                        state.reshape(1, state_update_interval, 6),
                        batch_size=1,
                        verbose=0
                    )

                    # Epsilon-greedy action selection
                    action = np.random.randint(0, 2) if random.random() < epsilon else np.argmax(q_value)

                    if take_action(action):
                        buffer.transition_detected()
                        clf_periodic_counter = 1
                        reward = get_reward(True, true_transitions, i)
                    else:
                        if clf_periodic_counter == clf_general_update_interval:
                            buffer.clf_train_general()
                            clf_periodic_counter = 1
                        else:
                            clf_periodic_counter += 1
                        buffer.clf_train_specific()
                        buffer.roll_window()
                        reward = get_reward(False, true_transitions, i)

                    # Update Q-values
                    reward = np.max(reward)
                    maxQ = np.max(q_value)
                    y = np.zeros((1, 2))
                    y[:] = q_value[:]
                    update = reward + (q_gamma * maxQ)
                    y[0][action] = update

                    # Train the agent
                    agent.fit(
                        state.reshape(1, state_update_interval, 6),
                        y,
                        batch_size=1,
                        epochs=1,
                        verbose=0
                    )

                    # Reset state and update reward
                    state[:, :] = 0.0
                    cumulative_reward[episode] = np.round(cumulative_reward[episode] + reward, 4)
                else:
                    state_update_counter += 1
                    if clf_periodic_counter == clf_general_update_interval:
                        buffer.clf_train_general()
                        clf_periodic_counter = 1
                    else:
                        clf_periodic_counter += 1
                    buffer.clf_train_specific()
                    buffer.roll_window()
        else:
            X, y = stream.next_sample()
            buffer.add_instance(X)
            i += 1

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon = round(epsilon - epsilon_decay, 4)

# Save the trained model
if save_model:
    print("Saving the trained model...")
    agent.save(f'trained_models/model_trained_{file_name}_new.h5')

# Save rewards if requested
if save_reward:
    df_cumulative_reward = pd.DataFrame({
        'episode': episode_number,
        'epsilon': epsilon_value,
        'reward': cumulative_reward
    })
    df_cumulative_reward.to_csv(f'output/cumulative_reward_{file_name}_new.csv', index=False)

# Calculate and save training time
elapsed = round((time.time() - start) / 60, 2)
print(f"\nTraining time: {elapsed} minutes")
