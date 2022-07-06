import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default=None, help='File dataset name.')
parser.add_argument('-s', '--sensitive', type=str, default=None, help='Sensitive column name.')
parser.add_argument('-l', '--label', type=str, default=None, help='Label column name.')
parser.add_argument('-p', '--percentage', type=float, default=70, help='Percentage of dataset train. From 0 to 100.')
parser.add_argument('-log', '--log_name', type=str, default='', help='File log name.')
parser.add_argument('-m', '--mutate', type=str, default=None, help='Mutate columns.')
args = parser.parse_args()

if not args.dataset or not args.sensitive or not args.label:
    print('Argumentos inválidos!')
    exit()

import logging

def createLog():
    logger = logging.getLogger(None)
    logger.setLevel(logging.INFO)
    if args.log_name: args.log_name = '-' + args.log_name
    fh = logging.FileHandler(args.dataset + args.log_name + '.log', "w")
    logger.addHandler(fh)

    return logger

logger = createLog()


import pandas as pd

dataset = pd.read_csv('datasets/' + args.dataset + '.csv')

logger.info(f'Dataset: {args.dataset}')
logger.info(f'Columns: {len(list(dataset.columns))} = {list(dataset.columns)}')
logger.info(f'Lines: {len(dataset)}')
qtd_label = dataset[f'{args.label}'].value_counts(sort=True)
qtd_sensitive = dataset[f'{args.sensitive}'].value_counts(sort=True)
logger.info(f'Number of Classes ({args.label}):\n{qtd_label}')
logger.info(f'Quantities in the Sensitive Attribute ({args.sensitive}):\n{qtd_sensitive}')


from sklearn.model_selection import train_test_split

def mutateColumns(x):
    if args.dataset == 'dutch-clean':
        mutant_economic_status = {111:-111, 112:-112, 120:-120}
        mutant_edu_level = {0:0, 1:1000, 2:2000, 3:3000, 4:4000, 5:5000}
        mutant_sex = {1:-10, 2:-20}
        x['economic_status'] = [mutant_economic_status[var] for var in x['economic_status']]
        x['edu_level'] = [mutant_edu_level[var] for var in x['edu_level']]
        # x['sex'] = [mutant_sex[var] for var in x['sex']]
    return x

x = dataset.drop([args.label], axis=1)
if args.mutate: x = mutateColumns(x)
y = dataset[args.label]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=args.percentage/100)

logger.info(f'\nRunning training with {args.percentage}% of data.')
logger.info(f'Train Lines: {len(x_train)}')
logger.info(f'Test Lines: {len(x_test)}')


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=60000)

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

cm = confusion_matrix_result = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
accuracy_balanced = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

logger.info(u'\nMetric Values of Logistic Regression Predictions:')
logger.info(f'Confusion Matrix: TN({cm[0][0]}) | FN({cm[1][0]}) | TP({cm[1][1]}) | FP({cm[0][1]})')
logger.info(f'Accuracy: {accuracy:.4f}')
logger.info(f'Balanced Accuracy: {accuracy_balanced:.4f}')
logger.info(f'Precision: {precision:.4f}')
logger.info(f'Recall: {recall:.4f}')
logger.info(f'F1-Score: {f1score:.4f}')


import matplotlib.pyplot as plt
import seaborn as sns

def graph_confusion_matrix(graph_title, confusion_matrix_result):
    plt.subplots(figsize=(9, 7))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues', fmt='d', cbar=False, annot_kws={"fontsize":15})
    plt.title(graph_title + '\n', loc='center', fontsize=15)
    plt.xlabel('\nPredict Values', fontsize=15)
    plt.ylabel('True Values\n', fontsize=15)
    plt.savefig(f'{args.dataset}-confusion-matrix.png')

graph_confusion_matrix(f'{args.dataset} predictions - confusion matrix', confusion_matrix_result)

def graph_bar(graph_title, metrics):
    labels = ['Accuracy',
              'Balanced\nAccuracy',
              'Precision',
              'Recall',
              'F1-Score',
              'Statistical\nParity',
              'Equalized\nOdds',
              'TPR Prot',
              'TPR Non-Prot',
              'TNR Prot',
              'TNR Non-Prot']
    plt.subplots(figsize=(9, 7))
    plt.title(graph_title + '\n', loc='center', fontsize=15)
    plt.barh(labels[:5], metrics[:5], height=.5)
    plt.barh(labels[5:], metrics[5:], height=.5)
    plt.gca().invert_yaxis()
    for index, value in enumerate(metrics):
        plt.text(x=value-0.1, y=index+0.1, s=f"{value:.4f}" , fontdict=dict(fontsize=11), color='white')
    plt.xlim(0.0,1.02)
    plt.savefig(f'{args.dataset}-metrics.png')





def calculate_performance(data, labels, predictions, saIndex, saValue):
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.

    tp_protected = 0.
    tn_protected = 0.
    fp_protected = 0.
    fn_protected = 0.

    tp_non_protected = 0.
    tn_non_protected = 0.
    fp_non_protected = 0.
    fn_non_protected = 0.
    for idx, val in enumerate(data):
        # protrcted population
        if val[saIndex] == saValue:
            if predictions[idx] == 1:
                protected_pos += 1.
            else:
                protected_neg += 1.


            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_protected += 1.
                else:
                    tn_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_protected += 1.
                else:
                    fp_protected += 1.

        else:
            if predictions[idx] == 1:
                non_protected_pos += 1.
            else:
                non_protected_neg += 1.

            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_non_protected += 1.
                else:
                    tn_non_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_non_protected += 1.
                else:
                    fp_non_protected += 1.

    tpr_protected = tp_protected / (tp_protected + fn_protected)
    tnr_protected = tn_protected / (tn_protected + fp_protected)

    tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
    tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)

    C_prot = (protected_pos) / (protected_pos + protected_neg)
    C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

    stat_par = C_non_prot - C_prot

    output = dict()

    # output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
    # output["balanced_accuracy"] =( (tp_protected + tp_non_protected)/(tp_protected + tp_non_protected + fn_protected + fn_non_protected) +
    #                                (tn_protected + tn_non_protected) / (tn_protected + tn_non_protected + fp_protected + fp_non_protected))*0.5

    # output["accuracy"] = accuracy_score(labels, predictions)
    # output["dTPR"] = tpr_non_protected - tpr_protected
    # output["dTNR"] = tnr_non_protected - tnr_protected
    # output["fairness"] = abs(tpr_non_protected - tpr_protected) + abs(tnr_non_protected - tnr_protected)
    # output["fairness"] = abs(stat_par)
    output['parity'] = stat_par
    output['equalized'] = abs(tpr_non_protected - tpr_protected) + abs(tnr_non_protected - tnr_protected)

    output["tpr_protected"] = tpr_protected
    output["tpr_non_protected"] = tpr_non_protected
    output["tnr_protected"] = tnr_protected
    output["tnr_non_protected"] = tnr_non_protected
    return output

sa_index = x_test.keys().tolist().index(args.sensitive)
p_group = 2

from fairlearn.metrics import *

# print(x_test[args.sensitive])

parity_difference = demographic_parity_difference(y_test, y_pred, sensitive_features=x_test[args.sensitive])
parity_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=x_test[args.sensitive])
equalized_difference = equalized_odds_difference(y_test, y_pred, sensitive_features=x_test[args.sensitive])
equalized_ratio = equalized_odds_ratio(y_test, y_pred, sensitive_features=x_test[args.sensitive])
tnr = true_negative_rate(y_test, y_pred)
tpr = true_positive_rate(y_test, y_pred)

logger.info(f'\nstatistical_parity: {parity_difference:.4f}')
logger.info(f'parity_ratio: {parity_ratio:.4f}')
logger.info(f'equalized_odds: {equalized_difference:.4f}')
logger.info(f'equalized_ratio: {equalized_ratio:.4f}')
logger.info(f'tnr: {tnr:.4f}')
logger.info(f'tpr: {tpr:.4f}')

manual_metrics = calculate_performance(x_test.values, y_test.values, y_pred, sa_index, p_group)
logger.info(f'\nStatistical Parity: {manual_metrics["parity"]:.4f}')
logger.info(f'Equalized Odds: {manual_metrics["equalized"]:.4f}')
logger.info(f'TPR Protected: {manual_metrics["tpr_protected"]:.4f}')
logger.info(f'TPR Non-Protected: {manual_metrics["tpr_non_protected"]:.4f}')
logger.info(f'TNR Protected: {manual_metrics["tnr_protected"]:.4f}')
logger.info(f'TNR Non-Protected: {manual_metrics["tnr_non_protected"]:.4f}')

graph_bar(f'{args.dataset} predictions - metrics', [accuracy,
                                                    accuracy_balanced,
                                                    precision,
                                                    recall,
                                                    f1score,
                                                    manual_metrics["parity"],
                                                    manual_metrics["equalized"],
                                                    manual_metrics["tpr_protected"],
                                                    manual_metrics["tpr_non_protected"],
                                                    manual_metrics["tnr_protected"],
                                                    manual_metrics["tnr_non_protected"]])