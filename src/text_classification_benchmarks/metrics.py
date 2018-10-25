from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import auc, confusion_matrix, precision_recall_fscore_support, roc_curve
from sklearn.preprocessing import label_binarize
from tabulate import tabulate
from typing import Union
import warnings

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def calc_true_positives(y_true, y_pred, k):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.logical_and(y_true == k, y_pred == k))


def calc_true_negatives(y_true, y_pred, k):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.logical_and(y_true != k, y_pred != k))


def calc_false_positives(y_true, y_pred, k):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.logical_and(y_true != k, y_pred == k))


def calc_false_negatives(y_true, y_pred, k):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.logical_and(y_true == k, y_pred != k))


def calc_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def calc_precision(y_true, y_pred, k):
    tp = calc_true_positives(y_true, y_pred, k)
    fp = calc_false_positives(y_true, y_pred, k)
    denom = tp + fp
    return 0 if denom == 0 else tp / denom


def calc_recall(y_true, y_pred, k):
    tp = calc_true_positives(y_true, y_pred, k)
    fn = calc_false_negatives(y_true, y_pred, k)
    denom = fn + tp
    return 0 if denom == 0 else tp / denom


def calc_f1_score(y_true, y_pred, k):
    """ Harmonic mean of precision and recall. """
    pre = calc_precision(y_true, y_pred, k)
    rec = calc_recall(y_true, y_pred, k)
    denom = pre + rec
    return 0 if denom == 0 else 2 * pre * rec / denom


def calc_support(y_true, k):
    return np.sum(y_true == k)


def calc_precision_macro_avg(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(y_true)
    pre_sum = 0
    for k in labels:
        pre_sum += calc_precision(y_true, y_pred, k)

    return pre_sum / len(labels)


def calc_precision_weighted_avg(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(y_true)
    counts = Counter(y_true)
    pre_sum = 0
    for k in labels:
        pre_sum += calc_precision(y_true, y_pred, k) * counts[k]

    return pre_sum / sum(counts.values())


def calc_precision_micro_avg(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(y_true)
    tp_sum = 0
    fp_sum = 0
    for k in labels:
        tp_sum += calc_true_positives(y_true, y_pred, k)
        fp_sum += calc_false_positives(y_true, y_pred, k)

    total_pred_positive = tp_sum + fp_sum
    return 0 if total_pred_positive == 0 else tp_sum / total_pred_positive


def calc_recall_macro_avg(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(y_true)
    rec_sum = 0
    for k in labels:
        rec_sum += calc_recall(y_true, y_pred, k)

    return rec_sum / len(labels)


def calc_recall_weighted_avg(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(y_true)
    counts = Counter(y_true)
    rec_sum = 0
    for k in labels:
        rec_sum += calc_recall(y_true, y_pred, k) * counts[k]

    return rec_sum / sum(counts.values())


def calc_recall_micro_avg(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(y_true)
    tp_sum = 0
    fn_sum = 0
    for k in labels:
        tp_sum += calc_true_positives(y_true, y_pred, k)
        fn_sum += calc_false_negatives(y_true, y_pred, k)

    total_actual_positive = fn_sum + tp_sum
    return 0 if total_actual_positive == 0 else tp_sum / total_actual_positive


def calc_f1_macro_avg(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(y_true)
    f1_sum = 0
    for k in labels:
        f1_sum += calc_f1_score(y_true, y_pred, k)

    return f1_sum / len(labels)


def calc_f1_weighted_avg(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(y_true)
    counts = Counter(y_true)
    f1_sum = 0
    for k in labels:
        f1_sum += calc_f1_score(y_true, y_pred, k) * counts[k]

    return f1_sum / sum(counts.values())


def calc_f1_micro_avg(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pre = calc_precision_micro_avg(y_true, y_pred)
    rec = calc_recall_micro_avg(y_true, y_pred)
    denom = pre + rec
    return 0 if denom == 0 else 2 * pre * rec / denom


def calc_multiclass_roc_auc(y_true, y_pred, average: Union[str, None]='macro'):
    labels = sorted(np.unique(y_true))
    n_classes = len(labels)
    y_true_bin = label_binarize(y_true, classes=labels)
    y_pred_bin = label_binarize(y_pred, classes=labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += sp.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    if average == 'macro':
        return roc_auc['macro']
    elif average == 'micro':
        return roc_auc['micro']
    else:
        # raise ValueError("Invalid argument 'average'. Expected values: ['macro', 'micro']")
        return roc_auc


def log_loss(y_true, y_pred, eps=1e-15):
    """ As used by Kaggle. """
    y_pred = sp.maximum(eps, y_pred)
    y_pred = sp.minimum(1 - eps, y_pred)
    ll = sum(y_true * sp.log(y_pred) + sp.subtract(1, y_true) * sp.log(sp.subtract(1, y_pred)))
    ll = ll * -1.0 / len(y_true)
    return ll


def calc_multiclass_log_loss(y_true, y_pred):
    scores = []
    for i in range(len(y_pred)):
        result = log_loss(y_true[i], y_pred[i])
        scores.append(result)

    return sum(scores) / len(scores)


def pad_name(name, pad_len):
    padded = [' '] * pad_len
    for i in range(min(len(name), pad_len)):
        padded[i] = name[i]

    return ''.join(padded)


def classification_report_to_df(y_true, y_pred, classes, counts_by_label, label_fixed_width=40):
    summary = precision_recall_fscore_support(y_true, y_pred, warn_for=())
    avg_tot = list(precision_recall_fscore_support(y_true, y_pred, average='weighted', warn_for=()))
    index = ['precision', 'recall', 'f1_score', 'support']
    df = pd.DataFrame(list(summary), index=index)
    support = df.loc['support']
    total = support.sum()
    avg_tot[-1] = total
    df[pad_name('Avg / Total', label_fixed_width)] = avg_tot
    df = df.T
    df['label'] = df.index
    df.ix[-1, 'label'] = ''
    df.rename(index={i: pad_name(classes[i], label_fixed_width)
                     for i in df.index.values[:-1]}, inplace=True)
    df['n_examples'] = df['label'].apply(lambda x: counts_by_label.get(x, 0))
    df.ix[-1, 'n_examples'] = df['n_examples'].sum()
    df = df[['label', 'precision', 'recall', 'f1_score', 'support', 'n_examples']]
    df = df.round({'precision': 2, 'recall': 2, 'f1_score': 2})
    return df


def perf_summary(y_true, y_pred):
    return {
        'precision_weighted_avg': calc_precision_weighted_avg(y_true, y_pred),
        'recall_weighted_avg': calc_recall_weighted_avg(y_true, y_pred),
        'f1_weighted_avg': calc_f1_weighted_avg(y_true, y_pred),
        'accuracy': calc_accuracy(y_true, y_pred),
        'roc_auc': calc_multiclass_roc_auc(y_true, y_pred)
    }


def print_perf_summary(stats, rounded=4):
    print('Precision (weighted avg):', round(stats['precision_weighted_avg'], rounded))
    print('Recall (weighted avg)   :', round(stats['recall_weighted_avg'], rounded))
    print('F1 Score (weighted avg) :', round(stats['f1_weighted_avg'], rounded))
    print('Accuracy                :', round(stats['accuracy'], rounded))
    print('ROC AUC (macro avg)     :', round(stats['roc_auc'], rounded))


def perf_by_label(y_true, y_pred, classes, counts_by_label):
    labels = sorted(np.unique(y_true))
    roc_auc = calc_multiclass_roc_auc(y_true, y_pred, average=None)
    rows = [['name', 'idx', 'precision', 'recall', 'f1_score', 'roc_auc', 'support', 'n_examples']]
    total_support = 0
    total_cases = 0
    for i, label in enumerate(labels):
        support = calc_support(y_true, label)
        total_support += support
        n_cases = counts_by_label[label]
        total_cases += n_cases
        rows.append([
            classes[label],
            label,
            calc_precision(y_true, y_pred, label),
            calc_recall(y_true, y_pred, label),
            calc_f1_score(y_true, y_pred, label),
            roc_auc[i],
            support,
            n_cases
        ])

    rows.append([
        'Avg / Total',
        '',
        calc_precision_weighted_avg(y_true, y_pred),
        calc_recall_weighted_avg(y_true, y_pred),
        calc_f1_weighted_avg(y_true, y_pred),
        calc_multiclass_roc_auc(y_true, y_pred),
        total_support,
        total_cases
    ])

    return rows


def print_perf_by_label(stats, rounded=4, sort_column='f1_score', ascending=False) -> pd.DataFrame:
    mat = np.array(stats)
    columns = mat[0]
    n_cols = mat.shape[1]
    data = {}
    for i in range(n_cols):
        if i > 5:
            data[columns[i]] = mat[1:, i].astype(int)
        elif i > 1:
            data[columns[i]] = np.round(mat[1:, i].astype(float), rounded)
        else:
            data[columns[i]] = mat[1:, i]

    df = pd.DataFrame(data, columns=columns)
    result: pd.DataFrame = pd.concat([df[:-2].sort_values(sort_column, ascending=ascending), df[-1:]])
    return result


def print_best_worst(stats, rounded=4, sort_column='f1_score', top_n=10, max_name_len=None):
    df = print_perf_by_label(stats, rounded, sort_column)
    df = df[df[sort_column] != 0]
    columns = ['name', 'idx', 'precision', 'recall', 'f1_score', 'roc_auc', 'support', 'n_examples']
    n_columns = len(columns)
    if max_name_len:
        df.name = df.name.apply(lambda x: pad_name(x, max_name_len))

    rows = [
        ['Best {}:'.format(top_n), *[''] * (n_columns - 1)],
        *df[:top_n].values.tolist(),
        [''] * n_columns,
        ['Worst {}:'.format(top_n), *[''] * (n_columns - 1)],
        *df[-(top_n + 1):-1].values.tolist(),
        [''] * n_columns,
        *df[-1:].values.tolist()
    ]
    print(tabulate(rows, headers=columns))


def plot_confusion_matrix(y_true, y_pred, classes, start, end):
    confusion_mat = confusion_matrix(y_true, y_pred)
    label_names_df = pd.DataFrame({'label': y_true, 'name': [classes[x] for x in y_true]})
    label_names_df = label_names_df.drop_duplicates().sort_values('label')
    # noinspection PyTypeChecker
    plt.subplots(figsize=(10, 10))
    sns.heatmap(confusion_mat[start:end, start:end], annot=True, fmt='d',
                xticklabels=label_names_df.name.values[start:end],
                yticklabels=label_names_df.name.values[start:end])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
