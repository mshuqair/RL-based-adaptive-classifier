from skmultiflow.data.data_stream import DataStream
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
import random
import time


# Classes for each anomaly detector
class StreamBufferOriginal:
    def __init__(self, w_size, w_dim, w_slide):
        self.size = w_size
        self.dim = w_dim
        self.slide = w_slide
        self.window_index = 0
        self.transition_count = 0
        self.start = True
        self.transition_triggered = False

    def at_start(self):
        if self.start:
            self.start = False
            return True
        else:
            return False

    def is_full(self):
        if self.window_index >= self.size:
            return True
        else:
            return False

    def add_instance(self):
        self.window_index = self.window_index + 1

    def roll_window(self):
        self.window_index = self.window_index - 1 * self.slide

    def transition_detected(self):
        self.transition_triggered = True
        self.transition_count = self.transition_count + 1
        self.window_index = 0
        self.start = True


class StreamBufferC1C2:
    def __init__(self, w_size, w_dim, w_slide):
        self.size = w_size
        self.dim = w_dim
        self.slide = w_slide
        self.win_data = np.zeros((self.size, self.dim))

        self.win_outliers_clf_c1 = np.zeros(self.size)
        self.win_outliers_clf_c1_percent = 1
        self.training_data_off = np.zeros((0, self.dim))
        self.clf_c1_trained = False

        self.win_outliers_clf_c2 = np.zeros(self.size)
        self.win_outliers_clf_c2_percent = 1
        self.training_data_on = np.zeros((0, self.dim))
        self.clf_c2_trained = False

        self.window_index = 0
        self.transition_count = 0
        if start_negative:  # False: start with clf_c1, True: to start with clf_c2
            self.clf_selector = 0
        else:
            self.clf_selector = 1
        self.start = True
        self.transition_triggered = False
        self.win_temp = np.zeros((self.size, self.dim))

        self.clf_c1 = OneClassSVM(nu=nu, kernel=kernel, degree=degree)
        self.clf_c2 = OneClassSVM(nu=nu, kernel=kernel, degree=degree)

    def at_start(self):
        if self.start:
            return True
        else:
            return False

    def is_full(self):
        if self.window_index >= self.size:
            return True
        else:
            return False

    def add_instance(self, data):
        self.win_data[self.window_index] = data
        if self.clf_selector == 0:
            self.training_data_off = np.append(self.training_data_off, data, axis=0)
        if self.clf_selector == 1:
            self.training_data_on = np.append(self.training_data_on, data, axis=0)
        self.window_index = self.window_index + 1

    def roll_window(self):
        self.win_data = np.roll(self.win_data, -1 * self.slide, axis=0)
        self.window_index = self.window_index - 1 * self.slide

    def score_check(self):
        if self.clf_c1_trained:
            self.win_outliers_clf_c1 = self.clf_c1.predict(self.win_data)
            temp, freq = np.unique(self.win_outliers_clf_c1, return_counts=True)
            self.win_outliers_clf_c1_percent = round(freq[0] / self.size, 4)

        if self.clf_c2_trained:
            self.win_outliers_clf_c2 = self.clf_c2.predict(self.win_data)
            temp, freq = np.unique(self.win_outliers_clf_c2, return_counts=True)
            self.win_outliers_clf_c2_percent = round(freq[0] / self.size, 4)

    def transition_detected(self):
        self.transition_triggered = True
        self.transition_count = self.transition_count + 1
        self.clf_selector = 1 - self.clf_selector  # to toggle between Class 1 and Class 2
        self.window_index = 0
        self.win_temp = self.win_data
        self.start = True

    def clf_train(self):
        if self.clf_selector == 0:
            self.clf_c1.fit(self.training_data_off)
            self.clf_c1_trained = True
        if self.clf_selector == 1:
            self.clf_c2.fit(self.training_data_on)
            self.clf_c2_trained = True


class StreamBufferGeneral:
    def __init__(self, w_size, w_dim, w_slide):
        self.size = w_size
        self.dim = w_dim
        self.slide = w_slide
        self.win_data = np.zeros((self.size, self.dim))
        self.win_outliers = np.zeros(self.size)
        self.win_outliers_percent = 0
        self.training_data = np.zeros((0, self.dim))
        self.window_index = 0
        self.transition_count = 0
        self.start = True
        self.transition_triggered = False
        self.clf = OneClassSVM(nu=nu, kernel=kernel, degree=degree)

    def at_start(self):
        if self.start:
            self.start = False
            return True
        else:
            return False

    def is_full(self):
        if self.window_index >= self.size:
            return True
        else:
            return False

    def add_instance(self, data):
        self.win_data[self.window_index] = data
        self.training_data = np.append(self.training_data, data, axis=0)
        self.window_index = self.window_index + 1

    def roll_window(self):
        self.win_data = np.roll(self.win_data, -1 * self.slide, axis=0)
        self.window_index = self.window_index - 1 * self.slide

    def score_check(self):
        self.win_outliers = self.clf.predict(self.win_data)
        temp, freq = np.unique(self.win_outliers, return_counts=True)
        self.win_outliers_percent = round(freq[0] / self.size, 4)

    def transition_detected(self):
        self.transition_triggered = True
        self.transition_count = self.transition_count + 1
        self.window_index = 0
        self.training_data = np.zeros((0, self.dim))
        self.start = True

    def clf_train(self):
        train_data = self.training_data
        self.clf.fit(train_data)


def build_agent():
    print("Building the agent...\n")
    model = Sequential()
    model.add(LSTM(32, input_shape=(state_update_interval, 6)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))
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
    periodic_slope = get_slope(temp_state[0, 0], buffer_general.win_outliers_percent)
    clf_c1_slope = get_slope(temp_state[0, 2], buffer_c1_c2.win_outliers_clf_c1_percent)
    clf_c2_slope = get_slope(temp_state[0, 4], buffer_c1_c2.win_outliers_clf_c2_percent)
    current_state = np.array([buffer_general.win_outliers_percent, periodic_slope,
                              buffer_c1_c2.win_outliers_clf_c1_percent, clf_c1_slope,
                              buffer_c1_c2.win_outliers_clf_c2_percent, clf_c2_slope])
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


def load_data():
    data = pd.read_csv('data/mhealth_subject5.csv')
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
df, true_transitions = load_data()
print("Actual label changes at: " + str(true_transitions))

# General parameters
file_name = 'mhealth_s5'  # name used for saving model, plots, reward, etc...
save_model = True  # to save the trained mode
save_train_time = False  # to save training time
save_reward = False  # to save the cumulative reward
plot_reward = True  # to plot the cumulative reward
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

# Initializing some parameters
state = np.zeros((state_update_interval, 6))
action = False
clf_periodic_counter = 1
state_update_counter = 1
cumulative_reward = np.zeros(episodes)
epsilon_value = np.zeros(episodes)
episode_number = np.zeros(episodes)

start = time.time()

for episode in range(episodes):
    print("Episode number: " + str(episode + 1))
    print("Epsilon value: " + str(epsilon))
    episode_number[episode] = episode + 1
    epsilon_value[episode] = epsilon

    stream = DataStream(df)
    buffer = StreamBufferOriginal(size, stream.n_features, slide)
    buffer_general = StreamBufferGeneral(size, stream.n_features, slide)
    buffer_c1_c2 = StreamBufferC1C2(size, stream.n_features, slide)

    i = int(0)
    while stream.has_more_samples():
        if buffer.is_full():
            buffer_c1_c2.transition_triggered = False
            if buffer.at_start():
                buffer_general.clf_train()
                buffer_general.start = False
                buffer_c1_c2.clf_train()
                buffer_c1_c2.start = False
            else:
                buffer_general.score_check()
                buffer_c1_c2.score_check()

                state = get_state(state)

                if state_update_counter == state_update_interval:
                    state_update_counter = 1
                    # to use with LSTM 1 x rows x columns
                    q_value = agent.predict(state.reshape(1, state_update_interval, 6), batch_size=1, verbose=0)
                    if random.random() < epsilon:
                        action = np.random.randint(0, 2)
                    else:
                        action = (np.argmax(q_value))

                    if take_action(action):
                        buffer.transition_detected()
                        buffer_general.transition_detected()
                        clf_periodic_counter = 1
                        buffer_c1_c2.transition_detected()
                        reward = get_reward(True, true_transitions, i)
                    else:
                        if clf_periodic_counter == clf_general_update_interval:
                            buffer_general.clf_train()
                            clf_periodic_counter = 1
                        else:
                            clf_periodic_counter = clf_periodic_counter + 1
                        buffer_c1_c2.clf_train()
                        buffer.roll_window()
                        buffer_general.roll_window()
                        buffer_c1_c2.roll_window()
                        reward = get_reward(False, true_transitions, i)
                    reward = np.max(reward)
                    maxQ = np.max(q_value)
                    y = np.zeros((1, 2))
                    y[:] = q_value[:]
                    update = reward + (q_gamma * maxQ)
                    y[0][action] = update
                    agent.fit(state.reshape(1, state_update_interval, 6), y, batch_size=1, epochs=1, verbose=0)
                    state[:, :] = 0.0
                    cumulative_reward[episode] = np.round(cumulative_reward[episode] + reward, 4)
                else:
                    state_update_counter = state_update_counter + 1
                    if clf_periodic_counter == clf_general_update_interval:
                        buffer_general.clf_train()
                        clf_periodic_counter = 1
                    else:
                        clf_periodic_counter = clf_periodic_counter + 1
                    buffer_c1_c2.clf_train()
                    buffer.roll_window()
                    buffer_general.roll_window()
                    buffer_c1_c2.roll_window()
        else:
            X, y = stream.next_sample()
            buffer.add_instance()
            buffer_general.add_instance(X)
            buffer_c1_c2.add_instance(X)
            i = i + 1
    if epsilon > epsilon_min:
        epsilon = round(epsilon - epsilon_decay, 4)

if save_model:
    print("Saving the trained model...")
    agent.save('trained_models/model_trained_' + str(file_name) + '.h5')

if save_reward:
    df_cumulative_reward = pd.DataFrame({'episode': episode_number, 'epsilon': epsilon_value,
                                         'reward': cumulative_reward})
    df_cumulative_reward.to_csv('output/cumulative_reward_' + str(file_name) + '.csv', index=False)

# Calculate and save the training time
elapsed = format(time.time() - start, '.2f')
text = ("Training time: " + str(round((time.time() - start) / 60, 2)) + " minutes")
print("\n" + text)
if save_train_time:
    file2write = open('training_time_' + str(file_name) + '.txt', 'w')
    file2write.write(text)
    file2write.close()
