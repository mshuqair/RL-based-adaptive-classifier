import numpy as np
from sklearn.svm import OneClassSVM


class StreamBuffer:
    def __init__(self, w_size, w_dim, w_slide, nu, kernel, degree, start_negative):
        self.size = w_size
        self.dim = w_dim
        self.slide = w_slide
        self.win_data = np.zeros((self.size, self.dim))

        # General classifier attributes
        self.win_outliers = np.zeros(self.size)
        self.win_outliers_percent = 0
        self.training_data = np.zeros((0, self.dim))
        self.clf = OneClassSVM(nu=nu, kernel=kernel, degree=degree)

        # Class-specific classifier attributes
        self.win_outliers_clf_c1 = np.zeros(self.size)
        self.win_outliers_clf_c1_percent = 1
        self.win_outliers_clf_c2 = np.zeros(self.size)
        self.win_outliers_clf_c2_percent = 1
        self.training_data_off = np.zeros((0, self.dim))
        self.training_data_on = np.zeros((0, self.dim))
        self.clf_c1_trained = False
        self.clf_c2_trained = False
        self.clf_c1 = OneClassSVM(nu=nu, kernel=kernel, degree=degree)
        self.clf_c2 = OneClassSVM(nu=nu, kernel=kernel, degree=degree)

        # Common state attributes
        self.window_index = 0
        self.transition_count = 0
        self.start = True
        self.transition_triggered = False

        # Class selector for C1/C2
        self.clf_selector = 0 if start_negative else 1

    def at_start(self):
        if self.start:
            self.start = False
            return True
        return False

    def is_full(self):
        return self.window_index >= self.size

    def add_instance(self, data):
        self.win_data[self.window_index] = data
        self.training_data = np.append(self.training_data, [data], axis=0)

        # Update class-specific training data
        if self.clf_selector == 0:
            self.training_data_off = np.append(self.training_data_off, [data], axis=0)
        else:
            self.training_data_on = np.append(self.training_data_on, [data], axis=0)

        self.window_index += 1

    def roll_window(self):
        self.win_data = np.roll(self.win_data, -1 * self.slide, axis=0)
        self.window_index = max(0, self.window_index - self.slide)

    def score_check(self):
        # General classifier check
        self.win_outliers = self.clf.predict(self.win_data)
        temp, freq = np.unique(self.win_outliers, return_counts=True)
        self.win_outliers_percent = round(freq[0] / self.size, 4)

        # Class-specific checks
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
        self.transition_count += 1
        self.window_index = 0
        self.start = True

        # Reset general classifier data
        self.training_data = np.zeros((0, self.dim))

        # Toggle class selector
        self.clf_selector = 1 - self.clf_selector

    def clf_train_general(self):
        # Train general classifier
        self.clf.fit(self.training_data)

    def clf_train_specific(self):
        # Train-class-specific classifiers
        if self.clf_selector == 0:
            self.clf_c1.fit(self.training_data_off)
            self.clf_c1_trained = True
        else:
            self.clf_c2.fit(self.training_data_on)
            self.clf_c2_trained = True
