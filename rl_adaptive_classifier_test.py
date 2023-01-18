from skmultiflow.data.data_stream import DataStream
from keras.models import load_model
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
        self.window_index = self.window_index - 1*self.slide

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
        self.win_data = np.roll(self.win_data, -1*self.slide, axis=0)
        self.window_index = self.window_index - 1*self.slide

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
        self.win_data = np.roll(self.win_data, -1*self.slide, axis=0)
        self.window_index = self.window_index - 1*self.slide

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


def get_slope(y1, y2):
    if state_update_counter == 1:
        slope = 0
    else:
        slope = (y2-y1)/(state_update_interval-1)
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
    temp_state[int(state_update_counter-1), :] = current_state
    return temp_state


def take_action(act):
    actions = np.array([[False], [True]])  # list of actions for the agent to select from
    new_action = actions[act][0]  # just to convert the array to a scalar value
    return new_action


def change_model_label(label):
    if label == 1:
        label = 2
    elif label == 2:
        label = 1
    return label


def load_data():
    data = pd.read_csv('data/mhealth_subject5.csv')
    transitions = find_true_transition(data)
    return data, transitions


def find_true_transition(data):
    actual_change = np.array([], dtype='int')
    for j in range(len(data)-1):
        if data.iloc[j+1, -1] - data.iloc[j, -1] != 0:  # Last column selected
            actual_change = np.append(actual_change, j+1)
    return actual_change


# Main Code

# Load data
df, true_transitions = load_data()
print("Ground truth label changes at: " + str(true_transitions))

# General parameters
model_path = 'trained_models/model_trained_mhealth_s5.h5'
file_name = 'mhealth_s5'
start_negative = True   # when the starting class is 1
display_plots = True    # to display resulting plots
save_plots = False      # to save the plots
font_size = 'xx-large'  # control the font size in the plots
f1_average = 'weighted' # control the calculation of the f1 average


# Data window parameters
size = 40  # sliding window size (X_w)
slide = 10  # sliding window step size

# Anomaly detector parameters
nu = 0.2            # nu value for the anomaly detector
kernel = 'rbf'     # kernel function
degree = 3          # degree of the poly function

# RL model parameters
state_update_interval = 4  # steps for the state to be updated before feeding it to agent (u)
agent = load_model(model_path, compile=False)   # loading the saved agent (keras model)


# General Anomaly detector parameters
clf_general_update_interval = 4


# Initializing some parameters
state = np.zeros((state_update_interval, 6))
action = False
clf_general_counter = 1
state_update_counter = 1
model_output = np.zeros(len(df), dtype=int)
outliers_percent = np.array([])
outliers_percent_general = np.array([])
outliers_percent_c1 = np.array([])
outliers_percent_c2 = np.array([])
outliers_percent_index = np.array([])
outliers_transitions = np.array([])
outliers_transitions_index = np.array([])
if start_negative:
    predicted_label = 1
else:
    predicted_label = 2


stream = DataStream(df)
buffer = StreamBufferOriginal(size, stream.n_features, slide)
buffer_general = StreamBufferGeneral(size, stream.n_features, slide)
buffer_c1_c2 = StreamBufferC1C2(size, stream.n_features, slide)

i = int(0)
while stream.has_more_samples():
    model_output[i] = predicted_label
    if buffer.is_full():
        buffer_c1_c2.transition_triggered = False
        if buffer.at_start():
            buffer_general.clf_train()
            buffer_c1_c2.start = False
            buffer_c1_c2.clf_train()
        else:
            buffer_general.score_check()
            buffer_c1_c2.score_check()
            outliers_percent_general = np.append(outliers_percent_general, buffer_general.win_outliers_percent)
            outliers_percent_c1 = np.append(outliers_percent_c1, buffer_c1_c2.win_outliers_clf_c1_percent)
            outliers_percent_c2 = np.append(outliers_percent_c2, buffer_c1_c2.win_outliers_clf_c2_percent)
            outliers_percent_index = np.append(outliers_percent_index, i)

            state = get_state(state)

            if state_update_counter == state_update_interval:
                state_update_counter = 1
                q_value = agent.predict(state.reshape(1, state_update_interval, 6), batch_size=1, verbose=0)
                action = (np.argmax(q_value))
                state[:, :] = 0.0

                if take_action(action):
                    print("Transition event detected at: {}".format(i))
                    predicted_label = change_model_label(predicted_label)
                    buffer.transition_detected()
                    buffer_general.transition_detected()
                    clf_general_counter = 1
                    buffer_c1_c2.transition_detected()
                    outliers_transitions = np.append(outliers_transitions, df.iloc[i, stream.n_features])
                    outliers_transitions_index = np.append(outliers_transitions_index, i)
                else:
                    if clf_general_counter == clf_general_update_interval:
                        buffer_general.clf_train()
                        clf_general_counter = 1
                    else:
                        clf_general_counter = clf_general_counter + 1
                    buffer_c1_c2.clf_train()
                    buffer.roll_window()
                    buffer_general.roll_window()
                    buffer_c1_c2.roll_window()
            else:
                state_update_counter = state_update_counter + 1
                if clf_general_counter == clf_general_update_interval:
                    buffer_general.clf_train()
                    clf_general_counter = 1
                else:
                    clf_general_counter = clf_general_counter + 1
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

print("Number of detected class label transitions: " + str(buffer.transition_count))

# Model metrics
model_accuracy = round(accuracy_score(stream.y, model_output)*100, 2)
tn, fp, fn, tp = confusion_matrix(stream.y, model_output).ravel()
model_sensitivity = round(tp / (fn + tp)*100, 2)
model_specificity = round(tn / (fp + tn)*100, 2)
model_f1score = round(f1_score(stream.y, model_output, average=f1_average)*100, 2)
model_auc = round(roc_auc_score(stream.y, model_output)*100, 2)
print("Model prediction accuracy: " + str(model_accuracy) + "%")
print("Model prediction sensitivity: " + str(model_sensitivity) + "%")
print("Model prediction specificity: " + str(model_specificity) + "%")
print("Model prediction F1 score: " + str(model_f1score) + "%")
print("Model prediction AUC: " + str(model_auc) + "%")


# Plot code
if display_plots:
    plt.figure()
    plt.title('Original data')
    plt.xlabel("Minutes", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylabel("Class label", fontsize=font_size)
    d = np.arange(0, len(df)) / 60
    plt.plot(d, stream.y, c='g')
    plt.show()

    plt.title('Predicted class transitions')
    plt.xlabel("Minutes", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylabel("Class label", fontsize=font_size)
    plt.yticks(np.array([1, 2]), ["1", "2"], fontsize=font_size)
    d = np.arange(0, len(df)) / 60
    plt.plot(d, stream.y, c='g')
    plt.scatter(np.array(outliers_transitions_index[:]) / 60, outliers_transitions[:], c='r', marker='o')
    plt.legend(("Class label", "Predicted transitions"), loc='center right', fontsize=font_size)
    plt.show()

    plt.title('Model Prediction for Subject: ')
    plt.xlabel("Minutes", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylabel("Class Label", fontsize=font_size)
    plt.yticks(np.array([1, 2]), ["1", "2"], fontsize=font_size)
    plt.plot(d, model_output)
    plt.show()

    plt.title('Anomaly scores')
    plt.xlabel("Minutes", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylabel("Outliers percentage", fontsize=font_size)
    plt.ylim(0, 1.1)
    plt.yticks(fontsize=font_size)
    plt.plot(np.array(outliers_percent_index[:]) / 60, outliers_percent_general, linestyle='dotted')
    plt.plot(np.array(outliers_percent_index[:]) / 60, outliers_percent_c1, linestyle='dotted')
    plt.plot(np.array(outliers_percent_index[:]) / 60, outliers_percent_c2, linestyle='dotted')
    for k in range(np.size(outliers_transitions_index)):
        plt.axvline(outliers_transitions_index[k] / 60, color='grey', linestyle='--', linewidth=1)
    plt.scatter(np.array(outliers_percent_index[:]) / 60, outliers_percent_general, marker='o')
    plt.scatter(np.array(outliers_percent_index[:]) / 60, outliers_percent_c1, marker='o')
    plt.scatter(np.array(outliers_percent_index[:]) / 60, outliers_percent_c2, marker='o')
    plt.show()
