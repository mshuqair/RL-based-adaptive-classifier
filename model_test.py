import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Any
import numpy as np
import pandas as pd
from keras.models import load_model
from utilities.stream_buffers import StreamBuffer
from utilities.data_stream import DataStream


@dataclass
class ModelConfig:
    """Configuration parameters for the model."""
    file_name: str = 'mhealth_p1'
    model_path: str = 'models/model_trained_mhealth_p5.h5'
    start_negative: bool = True  # when starting class is 1

    # Window parameters
    window_size: int = 40
    slide_size: int = 10

    # Anomaly detector parameters
    nu: float = 0.2
    kernel: str = 'rbf'
    degree: int = 3

    # RL model parameters
    state_update_interval: int = 4
    clf_general_update_interval: int = 4


class AdaptiveClassifier:
    """Main classifier implementation."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.state = np.zeros((config.state_update_interval, 6))
        self.state_update_counter = 1
        self.clf_general_counter = 1

        # Load model
        try:
            self.agent = load_model(config.model_path, compile=False)
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def get_slope(self, y1, y2):
        """Calculate a slope between two points."""
        if self.state_update_counter == 1:
            slope = 0
        else:
            slope = (y2 - y1) / (self.config.state_update_interval - 1)

        slope = round(max(slope, 0.001), 4)
        return slope

    def get_state(self, temp_state: np.ndarray, buffer: StreamBuffer) -> np.ndarray:
        """Update and return the current state."""
        periodic_slope = self.get_slope(temp_state[0, 0], buffer.win_outliers_percent)
        clf_c1_slope = self.get_slope(temp_state[0, 2], buffer.win_outliers_clf_c1_percent)
        clf_c2_slope = self.get_slope(temp_state[0, 4], buffer.win_outliers_clf_c2_percent)

        current_state = np.array([
            buffer.win_outliers_percent, periodic_slope,
            buffer.win_outliers_clf_c1_percent, clf_c1_slope,
            buffer.win_outliers_clf_c2_percent, clf_c2_slope
        ])

        temp_state[self.state_update_counter - 1, :] = current_state
        return temp_state

    @staticmethod
    def take_action(act: int) -> bool:
        """Convert model output to boolean action."""
        actions = np.array([False, True])
        action = bool(actions[act])
        return action

    @staticmethod
    def change_model_label(label: int) -> int:
        """Toggle between class labels 1 and 2."""
        new_label = 2 if label == 1 else 1
        return new_label

    def process_stream(self, data: pd.DataFrame) -> List[Any]:
        """Process the data stream and return model outputs."""
        stream = DataStream(data)
        buffer = StreamBuffer(
            self.config.window_size,
            stream.n_features,
            self.config.slide_size,
            self.config.nu,
            self.config.kernel,
            self.config.degree,
            self.config.start_negative
        )

        # Initialize tracking arrays
        model_output = np.zeros(len(data), dtype=int)
        outliers_data = {
            'general': [], 'c1': [], 'c2': [],
            'index': [], 'transitions': [], 'transitions_index': []
        }

        predicted_label = 1 if self.config.start_negative else 2

        # Process stream
        i = 0
        while stream.has_more_samples():
            model_output[i] = predicted_label

            if buffer.is_full():
                predicted_label, i = self._process_full_buffer(
                    buffer, stream, predicted_label, i, outliers_data, data
                )
            else:
                X, _ = stream.next_sample()
                buffer.add_instance(X)
                i += 1

        return self._prepare_output(stream.y, model_output, outliers_data)

    def _process_full_buffer(
            self, buffer: StreamBuffer, stream: DataStream,
            predicted_label: int, i: int, outliers_data: dict, data: pd.DataFrame
    ) -> Tuple[int, int]:
        """Process a full buffer and update state."""
        buffer.transition_triggered = False

        if buffer.at_start():
            buffer.clf_train_general()
            buffer.clf_train_specific()
        else:
            predicted_label = self._update_state_and_predict(
                buffer, predicted_label, i, outliers_data, data
            )

        return predicted_label, i

    def _update_state_and_predict(
            self, buffer: StreamBuffer, predicted_label: int,
            i: int, outliers_data: dict, data: pd.DataFrame
    ) -> int:
        """Update state and make predictions."""
        buffer.score_check()
        self._update_outliers_data(buffer, i, outliers_data)
        self.state = self.get_state(self.state, buffer)

        if self.state_update_counter == self.config.state_update_interval:
            predicted_label = self._make_prediction(
                buffer, predicted_label, i, outliers_data, data
            )
        else:
            self._update_counters_and_train(buffer)
            self.state_update_counter += 1

        return predicted_label

    def _make_prediction(
            self, buffer: StreamBuffer, predicted_label: int,
            i: int, outliers_data: dict, data: pd.DataFrame
    ) -> int:
        """Make a prediction and update the model state."""
        q_value = self.agent.predict(
            self.state.reshape(1, self.config.state_update_interval, 6),
            batch_size=1, verbose=0
        )
        action = np.argmax(q_value)
        self.state_update_counter = 1
        self.state.fill(0.0)

        if self.take_action(action):
            print(f"Transition event detected at: {i}")
            predicted_label = self.change_model_label(predicted_label)
            self._handle_transition(buffer, i, outliers_data, data)
        else:
            self._update_counters_and_train(buffer)

        return predicted_label

    def _update_outliers_data(
            self, buffer: StreamBuffer, i: int, outliers_data: dict
    ) -> None:
        """Update outlier tracking data."""
        outliers_data['general'].append(buffer.win_outliers_percent)
        outliers_data['c1'].append(buffer.win_outliers_clf_c1_percent)
        outliers_data['c2'].append(buffer.win_outliers_clf_c2_percent)
        outliers_data['index'].append(i)

    def _handle_transition(
            self, buffer: StreamBuffer, i: int,
            outliers_data: dict, data: pd.DataFrame
    ) -> None:
        """Handle transition event detection."""
        buffer.transition_detected()
        self.clf_general_counter = 1
        outliers_data['transitions'].append(data.iloc[i, -1])
        outliers_data['transitions_index'].append(i)

    def _update_counters_and_train(self, buffer: StreamBuffer) -> None:
        """Update counters and train classifiers."""
        if self.clf_general_counter == self.config.clf_general_update_interval:
            buffer.clf_train_general()
            self.clf_general_counter = 1
        else:
            self.clf_general_counter += 1

        buffer.clf_train_specific()
        buffer.roll_window()

    @staticmethod
    def _prepare_output(
            y_true: np.ndarray, y_predicted: np.ndarray, outliers_data: dict
    ) -> List[Any]:
        """Prepare the final output for saving."""
        return [
            y_true, y_predicted,
            outliers_data['transitions_index'],
            outliers_data['transitions'],
            outliers_data['index'],
            outliers_data['general'],
            outliers_data['c1'],
            outliers_data['c2']
        ]


def main():
    """Main execution function."""
    config = ModelConfig()

    # Load data
    data = pd.read_csv(Path('data') / f'{config.file_name}.csv')

    # Initialize and run classifier
    classifier = AdaptiveClassifier(config)
    model_output = classifier.process_stream(data)

    # Save results
    output_path = Path('output') / f'{config.file_name}_model_output.pkl'
    with open(output_path, 'wb') as file:
        pickle.dump(model_output, file)

    print("Processing completed successfully")


if __name__ == "__main__":
    main()
