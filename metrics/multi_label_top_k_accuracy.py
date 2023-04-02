# this code is top k accuracy for multilabel classification
import torch
import numpy as np
import unittest

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
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)