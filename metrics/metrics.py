import unittest
import numpy as np
from sklearn import metrics

def compute_mean_average_precision(y_true, y_pred_proba):
    average_precisions = []
    for i in range(y_true.shape[1]):
        if np.all(np.sum(y_true[:, i], axis=0) > 0) and np.all(np.sum(y_pred_proba[:, i], axis=0) > 0):
            average_precision = metrics.average_precision_score(y_true[:, i], y_pred_proba[:, i], average='macro', pos_label=1)
        else:
            average_precision = 0
        average_precisions.append(average_precision)
    mean_average_precision = np.mean(average_precisions)
    return mean_average_precision, average_precisions

class TestMeanAveragePrecision(unittest.TestCase):
    def test_compute_mean_average_precision(self):
        y_true = np.array([[1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1]])
        y_pred_proba = np.array([[0.6, 0.4, 0.7, 0.8, 0.3, 0.9], [0.6, 0.4, 0.7, 0.8, 0.3, 0.9]])
        mean_average_precision, average_precisions = compute_mean_average_precision(y_true, y_pred_proba)
        print("mean_average_precision: ", mean_average_precision)
        microAP = metrics.average_precision_score(y_true, y_pred_proba, average='micro')
        print("microAP: ", microAP)
        macroAP = metrics.average_precision_score(y_true, y_pred_proba, average='macro')
        print("macroAP: ", macroAP)
         
if __name__ == "__main__":
    unittest.main()