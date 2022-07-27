import argparse

# arguments to file execution
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default=None, help='File dataset name.')
parser.add_argument('-s', '--sensitive', type=str, default=None, help='Sensitive column name.')
parser.add_argument('-l', '--label', type=str, default=None, help='Label column name.')
parser.add_argument('-p', '--percentage', type=float, default=70, help='Percentage of dataset train. From 0 to 100.')
parser.add_argument('-log', '--log_name', type=str, default=None, help='File log name.')
parser.add_argument('-m', '--mutate', type=str, default=False, help='Mutate columns.')
parser.add_argument('-b', '--balanced', type=str, default='unbalanced', help='Balance train dataset.')
args = parser.parse_args()

if not args.dataset or not args.sensitive or not args.label:
    print('Invalid Arguments!')
    exit()

files_title = args.dataset + f'-{args.balanced}'
if args.mutate: files_title += '-mutated'

import logging

def createLog():
    logger = logging.getLogger(None)
    logger.setLevel(logging.INFO)
    if not args.log_name: args.log_name = files_title
    fh = logging.FileHandler(args.log_name + '.log', "w")
    logger.addHandler(fh)

    return logger

logger = createLog()


import pandas as pd

dataset = pd.read_csv('datasets/' + args.dataset + '.csv')

logger.info(f'Dataset: {args.dataset}')
logger.info(f'Columns: {len(list(dataset.columns))} = {list(dataset.columns)}')
logger.info(f'Rows: {len(dataset)}')
qtd_label = dataset[f'{args.label}'].value_counts(sort=True)
qtd_sensitive = dataset[f'{args.sensitive}'].value_counts(sort=True)
logger.info(f'Number of Classes ({args.label}):\n{qtd_label}')
logger.info(f'Quantities in the Sensitive Attribute ({args.sensitive}):\n{qtd_sensitive}')



from sklearn.model_selection import train_test_split

# mutation function
def mutateColumns(data_x):
    if args.dataset == 'dutch-clean':
        data_x['household_size'] = [var / 100000 for var in data_x['household_position']]
        data_x['edu_level'] = [var * 1000 for var in data_x['edu_level']]
        data_x['economic_status'] = [var * -1000 for var in data_x['economic_status']]
        data_x['cur_eco_activity'] = [var * 3.1415 for var in data_x['cur_eco_activity']]
    return data_x

# new dataset without label column
data_x = dataset.drop([args.label], axis=1)
# apply mutation in dataset
if args.mutate: data_x = mutateColumns(data_x)
# array with label column
data_y = dataset[args.label]


# balanced data with undersampling or oversampling
if args.balanced == 'undersampling':
    from imblearn.under_sampling import RandomUnderSampler
    under_sampler = RandomUnderSampler(sampling_strategy='not minority')
    data_x, data_y = under_sampler.fit_resample(data_x, data_y)
elif args.balanced == 'oversampling':
    from imblearn.over_sampling import RandomOverSampler
    over_sampler = RandomOverSampler(sampling_strategy='not majority')
    data_x, data_y = over_sampler.fit_resample(data_x, data_y)



# train and test data division
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size=args.percentage/100)

logger.info(f'\nRunning training with {args.percentage}% of data.')
logger.info(f'Train Lines: {len(train_x)}')
logger.info(f'Test Lines: {len(test_x)}')



from sklearn.linear_model import LogisticRegression
# create logistic regression model
lr = LogisticRegression(max_iter=60000)
# train execution
lr.fit(train_x, train_y)
# prediction execution
y_pred = lr.predict(test_x)



from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

# classification metrics calculation
c_matrix = confusion_matrix_result = confusion_matrix(test_y, y_pred)
accuracy = accuracy_score(test_y, y_pred)
accuracy_balanced = balanced_accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred)
recall = recall_score(test_y, y_pred)
f1score = f1_score(test_y, y_pred)

logger.info(u'\nMetric Values of Logistic Regression Predictions:')
logger.info(f'Confusion Matrix: TN({c_matrix[0][0]}) | FN({c_matrix[1][0]}) | TP({c_matrix[1][1]}) | FP({c_matrix[0][1]})')
logger.info(f'Accuracy: {accuracy:.4f}')
logger.info(f'Balanced Accuracy: {accuracy_balanced:.4f}')
logger.info(f'Precision: {precision:.4f}')
logger.info(f'Recall: {recall:.4f}')
logger.info(f'F1-Score: {f1score:.4f}')

# fairness metrics calculation
def fairness_metrics_manually(data, labels, predictions, saIndex, saValue):
    protected_pos = 0. ; protected_neg = 0.
    non_protected_pos = 0 ; non_protected_neg = 0.

    tp_protected = 0. ; tn_protected = 0. ; fp_protected = 0. ; fn_protected = 0.
    tp_non_protected = 0. ; tn_non_protected = 0. ; fp_non_protected = 0. ; fn_non_protected = 0.

    for idx, val in enumerate(data):
        # protected population
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

# sensitive attribute column index in test dataset
sa_index = test_x.keys().tolist().index(args.sensitive)
# protected group of sensitive attribute
protected_group = 2

from fairlearn.metrics import *

parity_difference = demographic_parity_difference(test_y, y_pred, sensitive_features=test_x[args.sensitive])
parity_ratio = demographic_parity_ratio(test_y, y_pred, sensitive_features=test_x[args.sensitive])
equalized_difference = equalized_odds_difference(test_y, y_pred, sensitive_features=test_x[args.sensitive])
equalized_ratio = equalized_odds_ratio(test_y, y_pred, sensitive_features=test_x[args.sensitive])
tnr = true_negative_rate(test_y, y_pred)
tpr = true_positive_rate(test_y, y_pred)

logger.info(f'\nstatistical_parity: {parity_difference:.4f}')
logger.info(f'parity_ratio: {parity_ratio:.4f}')
logger.info(f'equalized_odds: {equalized_difference:.4f}')
logger.info(f'equalized_ratio: {equalized_ratio:.4f}')
logger.info(f'tnr: {tnr:.4f}')
logger.info(f'tpr: {tpr:.4f}')

fairness_metrics_manual = fairness_metrics_manually(test_x.values, test_y.values, y_pred, sa_index, protected_group)
logger.info(f'\nStatistical Parity: {fairness_metrics_manual["parity"]:.4f}')
logger.info(f'Equalized Odds: {fairness_metrics_manual["equalized"]:.4f}')
logger.info(f'TPR Protected: {fairness_metrics_manual["tpr_protected"]:.4f}')
logger.info(f'TPR Non-Protected: {fairness_metrics_manual["tpr_non_protected"]:.4f}')
logger.info(f'TNR Protected: {fairness_metrics_manual["tnr_protected"]:.4f}')
logger.info(f'TNR Non-Protected: {fairness_metrics_manual["tnr_non_protected"]:.4f}')


# plot graphs
import matplotlib.pyplot as plt
import seaborn as sns

def graph_confusion_matrix(confusion_matrix_result):
    plt.subplots(figsize=(9, 7))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues', fmt='d', cbar=False, annot_kws={"fontsize":15})
    plt.title(files_title + ' - confusion matrix' + '\n', loc='center', fontsize=15)
    plt.xlabel('\nPredict Values', fontsize=15)
    plt.ylabel('True Values\n', fontsize=15)
    plt.savefig(f'{files_title}-confusion-matrix.png')

graph_confusion_matrix(confusion_matrix_result)

def graph_bar(metrics):
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
    plt.title(files_title + ' - metrics' + '\n', loc='center', fontsize=15)
    plt.barh(labels[:5], metrics[:5], height=.5)
    plt.barh(labels[5:], metrics[5:], height=.5)
    plt.gca().invert_yaxis()
    for index, value in enumerate(metrics):
        plt.text(x=value+0.02, y=index+0.1, s=f"{value:.4f}" , fontdict=dict(fontsize=11), color='black')
    plt.xlim(0.0,1.2)
    plt.savefig(f'{files_title}-metrics.png')

graph_bar([accuracy,
           accuracy_balanced,
           precision,
           recall,
           f1score,
           fairness_metrics_manual["parity"],
           fairness_metrics_manual["equalized"],
           fairness_metrics_manual["tpr_protected"],
           fairness_metrics_manual["tpr_non_protected"],
           fairness_metrics_manual["tnr_protected"],
           fairness_metrics_manual["tnr_non_protected"]])