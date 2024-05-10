# required packages
import numpy as np
import pandas as pd

# TASK 1
true = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
model_prob_pred = np.array([0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25])

# determine threshold and predicted values
threshold = 0.5
model_pred = np.where(model_prob_pred >= threshold, 1, 0)

# create a table
df = pd.DataFrame({"True Value": true,
                   "Model Probability Prediction": model_prob_pred,
                   "Model Prediction": model_pred})

# calculate metrics
TN = df[(df["True Value"] == 0) & (df["Model Prediction"] == 0)].shape[0]
FP = df[(df["True Value"] == 0) & (df["Model Prediction"] == 1)].shape[0]
FN = df[(df["True Value"] == 1) & (df["Model Prediction"] == 0)].shape[0]
TP = df[(df["True Value"] == 1) & (df["Model Prediction"] == 1)].shape[0]

# create confusion matrix
matrix = np.array([[TP, FN],
                   [FP, TN]])

confusion_matrix = pd.DataFrame(matrix, columns=["Predicted 1", "Predicted 0"],
                                index=["Actual 1", "Actual 0"])
"""
          Predicted 1  Predicted 0
Actual 1            5            1
Actual 0            1            3
"""

accuracy = (TN + TP)/(TN + FP + FN + TP)
recall = TP/(FN + TP)
precision = TP/(TP + FP)
f1Score = (2 * recall * precision)/(recall + precision)

# show results
report = pd.DataFrame({"Metrics": ["Accuracy", "Recall", "Precision", "F1 Score"],
                       "Results": [accuracy, recall, precision, f1Score]})

"""
     Metrics   Results
0   Accuracy  0.800000
1     Recall  0.833333
2  Precision  0.833333
3   F1 Score  0.833333
"""

# TASK 2
confusion_matrix = pd.DataFrame([[5, 5], [90, 900]], columns=["Predicted 1", "Predicted 0"],
                                index=["Actual 1", "Actual 0"])

# calculate metrics
TP = confusion_matrix.iloc[0, 0]
FN = confusion_matrix.iloc[0, 1]
FP = confusion_matrix.iloc[1, 0]
TN = confusion_matrix.iloc[1, 1]

accuracy = (TN + TP)/(TN + FP + FN + TP)
recall = TP/(FN + TP)
precision = TP/(TP + FP)
f1Score = (2 * recall * precision)/(recall + precision)

# show results
report = pd.DataFrame({"Metrics": ["Accuracy", "Recall", "Precision", "F1 Score"],
                       "Results": [accuracy, recall, precision, f1Score]})

"""
     Metrics   Results
0   Accuracy  0.905000
1     Recall  0.500000
2  Precision  0.052632
3   F1 Score  0.095238
"""

# According to the results, accuracy score is good but recall and precision scores are bad.
# When we examine the confusion matrix, 5 out of 10 actual 1 values are predicted as 1.
# And 5 out of 95 predicted 1 values are actual 1 values.
# This means that the model is not good enough to predict the actual values.
