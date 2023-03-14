import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from prettytable import PrettyTable

def subsetAccuracy(y_test, y_score):
    """
    The subset accuracy evaluates the fraction of correctly classified examples
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_score: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    subsetaccuracy : float
        Subset Accuracy of our model
    """
    subsetaccuracy = 0.0

    for i in range(y_test.shape[0]):
        same = True
        for j in range(y_test.shape[1]):
            if y_test[i,j] != y_score[i,j]:
                same = False
                break
        if same:
            subsetaccuracy += 1.0
    
    return subsetaccuracy/y_test.shape[0]

def precision(y_test, y_score):
    """
    Precision of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_score: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precision : float
        Precision of our model
    """
    precision = 0.0

    for i in range(y_test.shape[0]):
        intersection = 0.0
        hXi = 0.0
        for j in range(y_test.shape[1]):
            hXi = hXi + int(y_score[i,j])
            if int(y_test[i,j]) == 1 and int(y_score[i,j]) == 1:
                intersection += 1
            
        if hXi != 0:
            precision = precision + float(intersection/hXi)

    precision = float(precision/y_test.shape[0])
    return precision

def recall(y_test, y_pred):
    """
    Recall of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred : sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recall : float
        recall of our model
    """
    recall = 0.0

    for i in range(y_test.shape[0]):
        intersection = 0.0
        Yi = 0.0
        for j in range(y_test.shape[1]):
            Yi = Yi + int(y_test[i,j])

            if y_test[i,j] == 1 and int(y_pred[i,j]) == 1:
                intersection = intersection + 1
    
        if Yi != 0:
            recall = recall + float(intersection/Yi)    
    
    recall = float(recall/y_test.shape[0])
    return recall

def fbeta(y_test, y_pred, beta=1):
    """
    FBeta of our model
    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred : sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbeta : float
        fbeta of our model
    """
    pr = precision(y_test, y_pred)
    re = recall(y_test, y_pred)

    num = float((1+pow(beta,2))*pr*re)
    den = float(pow(beta,2)*pr + re)

    if den != 0:
        fbeta = num/den
    else:
        fbeta = 0.0
    return fbeta

def getExactMatchRatio(y_true, y_pred):
    return np.all(y_true == y_pred, axis=1).mean()

def getAccuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def get_Hamming_Loss(y_true, y_pred):
    temp=0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])

def getPrecision(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i])
    return temp/ y_true.shape[0]

def getRecall(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
    return temp/ y_true.shape[0]

def getF1score(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # topk index
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # 펼쳐서 expand 비교하기 广播
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # 有对的就行
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def pass_threshold(output, threshold_val):
    return output > torch.tensor(threshold_val).cuda()


def f2_score(output, target, mask=None):
    if mask is not None:
        if len(output.shape) == 2:
            mask = torch.BoolTensor([mask, ] * output.shape[0])
        output = output[mask]
        target = target[mask]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    # TP    predict 和 label 同时为1
    tp += ((output == 1) & (target.data == 1)).sum().float()
    # TN    predict 和 label 同时为0
    tn += ((output == 0) & (target.data == 0)).sum().float()
    # FN    predict 0 label 1
    fn += ((output == 0) & (target.data == 1)).sum().float()
    # FP    predict 1 label 0
    fp += ((output == 1) & (target.data == 0)).sum().float()

    acc = (tp + tn) / (tp + tn + fn + fp)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    # F1 = 2 * r * p / (r + p)c
    lamba_ = 2
    f2 = (1 + lamba_ ** 2) * (p * r / (lamba_ ** 2 * p + r))
    return f2


def compute_evaluation_metric(output, target, metrics=None, mask=None):
    if metrics is None:
        metrics = {'a'}
    if mask is not None:
        if len(output.shape) == 2:
            mask = torch.BoolTensor([mask, ] * output.shape[0])
        output = output[mask]
        target = target[mask]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    # TP    predict 和 label 同时为1
    tp += ((output == 1) & (target.data == 1)).sum().float()
    # TN    predict 和 label 同时为0
    tn += ((output == 0) & (target.data == 0)).sum().float()
    # FN    predict 0 label 1
    fn += ((output == 0) & (target.data == 1)).sum().float()
    # FP    predict 1 label 0
    fp += ((output == 1) & (target.data == 0)).sum().float()

    acc = None
    p = None
    r = None
    f2 = None
    if 'a' in metrics:
        acc = (tp + tn) / (tp + tn + fn + fp)
    if 'p' in metrics or 'f2' in metrics:
        # if tp == 0:
        #     p = 0
        # else:
        p = tp / (tp + fp)
    if 'r' in metrics or 'f2' in metrics:
        # if tp == 0:
        #     r = 0
        # else:
        r = tp / (tp + fn)
    # F1 = 2 * r * p / (r + p)

    if 'f2' in metrics:
        lamba_ = 2
        f2 = (1 + lamba_ ** 2) * (p * r / (lamba_ ** 2 * p + r))
    return tp, tn, fn, fp, acc, p, r, f2


def compute_evaluation_metric2(tp, tn, fn, fp, metrics=None):
    if metrics is None:
        metrics = {'a'}
    acc = None
    p = None
    r = None
    f2 = None
    if 'a' in metrics:
        acc = (tp + tn) / (tp + tn + fn + fp)
    if 'p' in metrics or 'f2' in metrics:
        # if tp == 0:
        #     p = 0
        # else:
        p = tp / (tp + fp)
    if 'r' in metrics or 'f2' in metrics:
        # if tp == 0:
        #     r = 0
        # else:
        r = tp / (tp + fn)
    # F1 = 2 * r * p / (r + p)

    if 'f2' in metrics:
        lamba_ = 2
        f2 = (1 + lamba_ ** 2) * (p * r / (lamba_ ** 2 * p + r))
    return acc, p, r, f2

def fbeta(true_label, prediction):
    return metrics.fbeta_score(true_label, prediction, beta=2, average='samples')

def multi_label_confusion_matrix(y_true, y_pred, labels=None):
    matrix = metrics.multilabel_confusion_matrix(y_true, y_pred)
    if not labels:
        return matrix
    return show_multi_label_confusion_matrix(matrix, labels)

def multi_label_score_confusion_matrix(y_true, y_pred, num_labels, labels=None):
    scores_true, scores_pred = multi_label_score(y_true), multi_label_score(y_pred)
    matrix = metrics.confusion_matrix(scores_true, scores_pred, labels=range(num_labels))
    if not labels:
        return matrix
    return show_confusion_matrix(matrix, labels)

def show_multi_label_confusion_matrix(matrix, labels):
    result = ''
    for i, label in enumerate(labels):
        table = PrettyTable(['', 'not ' + label, label])
        table.add_row(['not ' + label, *[str(value) for value in matrix[i][0]]])
        table.add_row([label, *[str(value) for value in matrix[i][1]]])
        result += '\n' + str(table)
    return result

def show_confusion_matrix(matrix, labels):
    table = PrettyTable()
    indexes = []

    # find the indexes of  rows required
    for i in range(len(labels)):
        rows, columns = matrix[:, i], matrix[i, :]
        if any(columns) or any(rows):
            indexes.append(i)
            continue
    # remove the rows useless
    matrix = np.delete(matrix, [i for i in range(len(labels)) if i not in indexes], 0)

    # add the name of row
    table_columns = [('', [labels[i] for i in indexes])]
    # add the values
    for i in indexes:
        columns = matrix[:, i]
        table_columns.append((labels[i], columns))
    # show
    for label, columns in table_columns:
        table.add_column(label, columns)
    return str(table)

def multi_label_score(y):
    scores = None
    for y_true_one in y:
        indexes = np.argwhere(y_true_one)
        indexes_sum = indexes.sum()
        if indexes.shape[0] > 1:
            indexes_sum += indexes.shape[0]
        score = np.expand_dims(indexes_sum, 0)
        if scores is not None:
            scores = np.concatenate((scores, score), axis=0)
        else:
            scores = score
    return scores

def plotMultiROCCurve(y_true, y_score):
    """
    Plot the ROC curve
    Params
    ======
    y_true : sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    y_score: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    None
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(y_true.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    for i in range(y_true.shape[1]):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve : class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)
     
    # plt.plot(fpr[2], tpr[2], color='darkorange',
                # lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plotMultilabelconfusionmatrix(y_test, y_pred, labels):
    cm = metrics.multilabel_confusion_matrix(y_test, y_pred)
    ''' plot n * 4 subplots '''
    nClasses = len(labels)
    fig, ax = plt.subplots(int(nClasses/5), 5, figsize=(10, 8))
    for axes, cfs_matrix, label in zip(ax.flatten(), cm, labels):
        df_cm = pd.DataFrame(cfs_matrix, index = [i for i in ["True", "False"]],
                  columns = [i for i in ["True", "False"]])
        sns.heatmap(df_cm, annot=True, ax = axes, fmt='g')
        axes.set_title(label)
    fig.tight_layout()
    plt.show()

def plotMultilabelconfusionmatrix(y_test, y_pred, labels):
    cm = metrics.multilabel_confusion_matrix(y_test, y_pred)
    cm = metrics.multilabel_confusion_matrix(y_test, y_pred, labels=[0, 1])
    ''' plot n * 4 subplots '''
    nClasses = len(labels)
    fig, ax = plt.subplots(int(nClasses/5), 5, figsize=(10, 8))
    for axes, cfs_matrix, label in zip(ax.flatten(), cm, labels):
        df_cm = pd.DataFrame(cfs_matrix, index = [i for i in ["True", "False"]],
                  columns = [i for i in ["True", "False"]])
        sns.heatmap(df_cm, annot=True, ax = axes, fmt='g')
        axes.set_title(label)
    fig.tight_layout()
    plt.show()