import unittest
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def compute_mean_average_precision(y_true, y_pred_proba):
    average_precisions = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) == 1:
            average_precision = 1.0/(y_true.shape[1])
        else:
            average_precision = average_precision_score(y_true[:, i], y_pred_proba[:, i], average='macro', pos_label=1)
        average_precisions.append(average_precision)
    average_precisions = np.array(average_precisions)
    mean_average_precision = np.mean(average_precisions)
    return mean_average_precision, average_precisions

def compute_mean_roc_auc(y_true, y_pred_proba):
    roc_aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) == 1:
            roc_auc = 1.0 / (y_true.shape[1])
        else:
            roc_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        roc_aucs.append(roc_auc)
    roc_aucs = np.array(roc_aucs)
    mean_roc_auc = np.mean(roc_aucs)
    return mean_roc_auc, roc_aucs

def top_k_accuracy(y_true, y_pred_proba, k=5):
    """
    Calculate top-k accuracy.

    :param y_true: (list or numpy array) Ground truth (correct) labels.
    :param y_pred_proba: (list or numpy array) Predicted probabilities for each class.
    :param k: (int) Number of top predictions to consider.

    :return: (float) Top-k accuracy.
    """
    assert len(y_true) == len(y_pred_proba), "Input arrays must have the same length."
    assert k > 0, "k must be greater than 0."
    
    y_true = np.asarray(y_true)
    print(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Get the top-k predicted class indices for each sample
    top_k_preds = np.argsort(y_pred_proba, axis=-1)[:, -k:]
    print(top_k_preds)
    
    # Check if the true class is in the top-k predictions for each sample
    correct = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    
    # Calculate top-k accuracy
    top_k_acc = np.mean(correct)

    return top_k_acc

# this code is top k accuracy for multilabel classification
def multi_label_top_k_accuracy(y_true, y_pred_proba, k=5):
    """
    Calculate top-k accuracy for multi-label classification.

    :param y_true: (list or numpy array) Ground truth (correct) labels. Shape should be (n_samples, n_classes).
    :param y_pred_proba: (list or numpy array) Predicted probabilities for each class. Shape should be (n_samples, n_classes).
    :param k: (int) Number of top predictions to consider.

    :return: (float) Top-k accuracy.
    """
    assert y_true.shape == y_pred_proba.shape, "Input arrays must have the same shape."
    assert k > 0, "k must be greater than 0."
    
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Get the top-k predicted class indices for each sample
    top_k_preds = np.argsort(y_pred_proba, axis=-1)[:, -k:]
    
    # Check if all true labels are in the top-k predictions for each sample
    correct = [set(np.where(y_true[i] == 1)[0]).issubset(set(top_k_preds[i])) for i in range(len(y_true))]
    
    # Calculate top-k accuracy
    top_k_acc = np.mean(correct)

    return top_k_acc


def multi_label_top_margin_k_accuracy(y_true, y_pred_proba, margin=1):
    """
    Calculate top-k accuracy for multi-label classification, with k being the number of true labels plus margin.

    :param y_true: (list or numpy array) Ground truth (correct) labels. Shape should be (n_samples, n_classes).
    :param y_pred_proba: (list or numpy array) Predicted probabilities for each class. Shape should be (n_samples, n_classes).
    :param margin: (int) Value to add to the number of true labels to determine k.

    :return: (float) Top-k accuracy.
    """
    assert y_true.shape == y_pred_proba.shape, "Input arrays must have the same shape."
    
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    correct = []
    for i in range(len(y_true)):
        # Calculate k based on the number of true labels plus margin
        k = int(np.sum(y_true[i])) + margin
        
        # "k must be greater than 0 and less than or equal to the number of classes."
        assert (k > 0 and k <= y_true.shape[1]) 
        # Get the top-k predicted class indices for the sample
        top_k_preds = np.argsort(y_pred_proba[i])[-k:]
        
        # Check if all true labels are in the top-k predictions for the sample
        is_correct = set(np.where(y_true[i] == 1)[0]).issubset(set(top_k_preds))
        correct.append(is_correct)
    
    # Calculate top-k accuracy
    top_k_acc = np.mean(correct)

    return top_k_acc

class TestMeanAveragePrecision(unittest.TestCase):
    def test_compute_mean_average_precision(self):
        y_true = np.array([[1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1]])
        y_pred_proba = np.array([[0.6, 0.4, 0.7, 0.8, 0.3, 0.9], [0.6, 0.4, 0.7, 0.8, 0.3, 0.9]])
        mean_average_precision, average_precisions = compute_mean_average_precision(y_true, y_pred_proba)
        print("mean_average_precision: ", mean_average_precision)
        microAP = average_precision_score(y_true, y_pred_proba, average='micro')
        print("microAP: ", microAP)
        macroAP = average_precision_score(y_true, y_pred_proba, average='macro')
        print("macroAP: ", macroAP)
        mean_roc_auc, roc_aucs = compute_mean_roc_auc(y_true, y_pred_proba)
        print("mean_roc_auc: ", mean_roc_auc)
        
class test_top_k_accuracy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_top_k_accuracy, self).__init__(*args, **kwargs)
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def test_top_k_accuracy(self):
        y_true = np.zeros((10, 20))
        for i in range(10):
            y_true[i, np.random.choice(20, 1, replace=False)] = 1
        y_true_argmax = np.argmax(y_true, axis=-1) # only 1 true label per sample
        y_pred_proba = np.random.rand(10, 20)
        print(y_true_argmax)
        print(y_pred_proba)
        accuracy = top_k_accuracy(y_true_argmax, y_pred_proba, k=5)
        print("top_k_accuracy: ", accuracy)

    def test_multi_label_top_k_accuracy(self):
        y_true = np.zeros((10, 20))
        for i in range(10):
            y_true[i, np.random.choice(20, 2, replace=False)] = 1
        y_pred_proba = np.random.rand(10, 20)
        accuracy = multi_label_top_k_accuracy(y_true, y_pred_proba, k=5)
        print("multi_label_top_k_accuracy: ", accuracy)

    def test_multi_label_top_margin_k_accuracy(self):
        y_true = np.zeros((10, 20))
        for i in range(10):
            y_true[i, np.random.choice(20, 2, replace=False)] = 1
        y_pred_proba = np.random.rand(10, 20)
        accuracy = multi_label_top_margin_k_accuracy(y_true, y_pred_proba, margin=5)
        print("multi_label_top_margin_k_accuracy: ", accuracy)
        
         
if __name__ == "__main__":
    unittest.main()