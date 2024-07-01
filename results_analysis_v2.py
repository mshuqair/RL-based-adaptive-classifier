import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


# Main code
f1_average = 'weighted'
figure_size = (8, 5)
save_figure = True

# Load the data
with open('output/model_output.pkl', 'rb') as file:
    [y_true, y_predicted,
     class_transitions_index, class_transitions,
     outliers_percent_index, outliers_percent_general,
     outliers_percent_c1, outliers_percent_c2] = pickle.load(file)


# Metrics
accuracy = accuracy_score(y_true, y_predicted)
tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
sensitivity = tp / (fn + tp)
specificity = tn / (fp + tn)
f1score = f1_score(y_true, y_predicted, average=f1_average)
auc = roc_auc_score(y_true, y_predicted)
print('Model metrics:')
print('Accuracy %.2f, Sensitivity %.2f, Specificity %.2f, F1-score %.2f, AUC %.2f'
      % (accuracy, sensitivity, specificity, f1score, auc))


# Plot code
d = np.arange(0, y_true.shape[0]) / 60

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figure_size, layout='constrained')
ax[0, 0].set_title('Ground truth')
ax[0, 0].plot(d, y_true, c='g')
ax[0, 0].set_yticks(np.array([1, 2]), labels=[1, 2])

ax[0, 1].set_title('Predicted class transitions')
ax[0, 1].plot(d, y_true, c='g', label='Ground truth')
ax[0, 1].scatter(np.array(class_transitions_index[:]) / 60, class_transitions[:],
                 color='r', marker='o', label='Predicted transitions')
ax[0, 1].set_yticks(np.array([1, 2]), labels=[1, 2])
ax[0, 1].legend(loc='best', fontsize='small')


ax[1, 0].set_title('Predicted label')
ax[1, 0].plot(d, y_predicted)
ax[1, 0].set_xlabel(xlabel="Minutes")
ax[1, 0].set_yticks(np.array([1, 2]), labels=[1, 2])


ax[1, 1].set_title('Anomaly scores')
ax[1, 1].plot(np.array(outliers_percent_index[:]) / 60, outliers_percent_general,
              linestyle='dotted', marker='o', label='General')
ax[1, 1].plot(np.array(outliers_percent_index[:]) / 60, outliers_percent_c1,
              linestyle='dotted', marker='o', label='Class 1')
ax[1, 1].plot(np.array(outliers_percent_index[:]) / 60, outliers_percent_c2,
              linestyle='dotted', marker='o', label='Class 2')
for k in range(np.size(class_transitions_index)):
    ax[1, 1].axvline(class_transitions_index[k] / 60, color='grey', linestyle='--', linewidth=1)
ax[1, 1].set_xlabel(xlabel="Minutes")
ax[1, 1].set_ylim(0, 1.1)
ax[1, 1].legend(loc='best', fontsize='small')


if save_figure:
    plt.savefig('output/model_output.png')
plt.show()
