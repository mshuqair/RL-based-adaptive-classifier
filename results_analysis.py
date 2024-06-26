import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


# Main code
f1_average = 'weighted'
figure_size = (6, 3.5)

# Load the data
with open('output/model_output.pkl', 'rb') as file:
    [y_true, y_predicted,
     outliers_transitions_index, outliers_transitions,
     outliers_percent_index, outliers_percent_general,
     outliers_percent_c1, outliers_percent_c2] = pickle.load(file)


# Metrics
model_accuracy = round(accuracy_score(y_true, y_predicted) * 100, 2)
tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
model_sensitivity = round(tp / (fn + tp) * 100, 2)
model_specificity = round(tn / (fp + tn) * 100, 2)
model_f1score = round(f1_score(y_true, y_predicted, average=f1_average) * 100, 2)
model_auc = round(roc_auc_score(y_true, y_predicted) * 100, 2)
print("Model prediction accuracy: " + str(model_accuracy) + "%")
print("Model prediction sensitivity: " + str(model_sensitivity) + "%")
print("Model prediction specificity: " + str(model_specificity) + "%")
print("Model prediction F1 score: " + str(model_f1score) + "%")
print("Model prediction AUC: " + str(model_auc) + "%")

# Plot code
d = np.arange(0, y_true.shape[0]) / 60

plt.figure(figsize=figure_size)
plt.title('Original data')
plt.xlabel(xlabel="Minutes")
plt.ylabel(ylabel="Class label")
plt.yticks(np.array([1, 2]), labels=["1", "2"])
plt.plot(d, y_true, c='g')
plt.tight_layout()
plt.show()

plt.figure(figsize=figure_size)
plt.title('Predicted class transitions')
plt.xlabel(xlabel="Minutes")
plt.ylabel(ylabel="Class label")
plt.yticks(np.array([1, 2]), labels=["1", "2"])
plt.plot(d, y_true, c='g')
plt.scatter(np.array(outliers_transitions_index[:]) / 60, outliers_transitions[:], c='r', marker='o')
plt.legend(("Class label", "Predicted transitions"), loc='center right')
plt.tight_layout()
plt.show()

plt.figure(figsize=figure_size)
plt.title('Model Prediction')
plt.xlabel("Minutes")
plt.ylabel("Class Label")
plt.yticks(np.array([1, 2]), labels=["1", "2"])
plt.plot(d, y_predicted)
plt.tight_layout()
plt.show()

plt.figure(figsize=figure_size)
plt.title('Anomaly scores')
plt.xlabel(xlabel="Minutes")
plt.ylabel(ylabel="Outliers percentage")
plt.ylim(0, 1.1)
plt.plot(np.array(outliers_percent_index[:]) / 60, outliers_percent_general, linestyle='dotted')
plt.plot(np.array(outliers_percent_index[:]) / 60, outliers_percent_c1, linestyle='dotted')
plt.plot(np.array(outliers_percent_index[:]) / 60, outliers_percent_c2, linestyle='dotted')
for k in range(np.size(outliers_transitions_index)):
    plt.axvline(outliers_transitions_index[k] / 60, color='grey', linestyle='--', linewidth=1)
plt.scatter(np.array(outliers_percent_index[:]) / 60, outliers_percent_general, marker='o')
plt.scatter(np.array(outliers_percent_index[:]) / 60, outliers_percent_c1, marker='o')
plt.scatter(np.array(outliers_percent_index[:]) / 60, outliers_percent_c2, marker='o')
plt.tight_layout()
plt.show()
