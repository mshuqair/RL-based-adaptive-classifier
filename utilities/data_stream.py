

class DataStream:
    def __init__(self, data):
        data = data.values
        self.X = data[:, :-1]
        self.y = data[:, -1].astype(int)
        self.index = 0
        self.n_features = self.X.shape[1]

    def has_more_samples(self):
        return self.index < len(self.X)

    def next_sample(self):
        sample = self.X[self.index, :]
        label = self.y[self.index]
        self.index += 1
        return sample, label
