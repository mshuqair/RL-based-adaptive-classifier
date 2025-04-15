import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# Configuration
f1_average = "weighted"
figure_size = (6, 3.5)
save_figure = False

# Load the model output
with open("output/model_output.pkl", "rb") as file:
    (
        y_true,
        y_predicted,
        class_transitions_index,
        class_transitions,
        outliers_percent_index,
        outliers_percent_general,
        outliers_percent_c1,
        outliers_percent_c2,
    ) = pickle.load(file)

# Compute metrics
accuracy = accuracy_score(y_true, y_predicted)
tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
sensitivity = tp / (fn + tp)
specificity = tn / (fp + tn)
f1score = f1_score(y_true, y_predicted, average=f1_average)
auc = roc_auc_score(y_true, y_predicted)

print("Model metrics:")
print(
    "Accuracy: %.2f | Sensitivity: %.2f | Specificity: %.2f | F1-score: %.2f | AUC: %.2f"
    % (accuracy, sensitivity, specificity, f1score, auc)
)

# Time in minutes
d = np.arange(0, y_true.shape[0]) / 60

# Plot: Original Data (Ground Truth)
plt.figure(figsize=figure_size)
plt.title("Original data")
plt.xlabel("Minutes")
plt.yticks([1, 2], labels=["Non-walking", "Walking"])
plt.plot(d, y_true, c="g")
plt.tight_layout()
if save_figure:
    plt.savefig("output/ground_truth.png")
plt.show()

# Plot: Predicted Class Transitions
plt.figure(figsize=figure_size)
plt.title("Predicted class transitions")
plt.xlabel("Minutes")
plt.yticks([1, 2], labels=["Non-walking", "Walking"])
plt.plot(d, y_true, c="g")
plt.scatter(
    np.array(class_transitions_index) / 60,
    class_transitions,
    c="r",
    marker="o",
)
plt.legend(("Class label", "Predicted transitions"), loc="center right")
plt.tight_layout()
if save_figure:
    plt.savefig("output/predicted_class_transition.png")
plt.show()

# Plot: Predicted Output
plt.figure(figsize=figure_size)
plt.title("Model Prediction")
plt.xlabel("Minutes")
plt.yticks([1, 2], labels=["Non-walking", "Walking"])
plt.plot(d, y_predicted)
plt.tight_layout()
if save_figure:
    plt.savefig("output/predicted_output.png")
plt.show()

# Plot: Anomaly Scores
plt.figure(figsize=figure_size)
plt.title("Anomaly scores")
plt.xlabel("Minutes")
plt.ylabel("Outliers percentage")
plt.ylim(0, 1.1)

# Plot each anomaly line
plt.plot(
    np.array(outliers_percent_index) / 60,
    outliers_percent_general,
    linestyle="dotted",
)
plt.plot(
    np.array(outliers_percent_index) / 60,
    outliers_percent_c1,
    linestyle="dotted",
)
plt.plot(
    np.array(outliers_percent_index) / 60,
    outliers_percent_c2,
    linestyle="dotted",
)

# Add vertical lines for transitions
for idx in class_transitions_index:
    plt.axvline(idx / 60, color="grey", linestyle="--", linewidth=1)

# Add scatter points
plt.scatter(
    np.array(outliers_percent_index) / 60,
    outliers_percent_general,
    marker="o",
)
plt.scatter(
    np.array(outliers_percent_index) / 60,
    outliers_percent_c1,
    marker="o",
)
plt.scatter(
    np.array(outliers_percent_index) / 60,
    outliers_percent_c2,
    marker="o",
)

plt.tight_layout()
if save_figure:
    plt.savefig("output/anomaly_scores.png")
plt.show()
