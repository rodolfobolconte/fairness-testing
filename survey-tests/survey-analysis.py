# passar dataset
# retorna:
#   txt com métricas
#   png dos gráficos

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='', help='File dataset name.')
parser.add_argument('-s', '--sensitive', type=str, default='', help='Sensitive column name.')
parser.add_argument('-l', '--label', type=str, default='', help='Label column name.')
parser.add_argument('-p', '--percentage', type=float, default=70, help='Percentage of dataset train. From 0 to 100.')
args = parser.parse_args()


import logging

def createLog():
    logger = logging.getLogger(None)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(args.dataset + '.log', "w")
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
logger.info(f'Quantities of Classes ({args.label}):\n{qtd_label}')
logger.info(f'Quantities in the Sensitive Attribute ({args.sensitive}):\n{qtd_sensitive}')


from sklearn.model_selection import train_test_split

x = dataset.drop([args.label], axis=1)
y = dataset[args.label]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=args.percentage/100)

logger.info(f'\nRunning training with {args.percentage}% of data.')
logger.info(f'Train Lines: {len(x_train)}')
logger.info(f'Test Lines: {len(x_test)}')


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=60000)

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

cm = confusion_matrix_result = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

logger.info(u'\nMetric Values of Logistic Regression Predictions:')
logger.info(f'Confusion Matrix: TN({cm[0][0]}) | FN({cm[1][0]}) | TP({cm[1][1]}) | FP({cm[0][1]})')
logger.info(f'Accuracy: {accuracy:.4f}')
logger.info(f'Precision: {precision:.4f}')
logger.info(f'Recall: {recall:.4f}')

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

def graph_bar(graph_title, metrics_prevision):
    labels = ['Accuracy', 'Precision', 'Recall']
    plt.subplots(figsize=(9, 7))
    plt.title(graph_title + '\n', loc='center', fontsize=15)
    plt.bar(labels, [metrics_prevision[0], metrics_prevision[1], metrics_prevision[2]], width=.5)
    for index, data in enumerate([metrics_prevision[0], metrics_prevision[1], metrics_prevision[2]]):
        plt.text(x=index-0.1, y=data-0.02, s=f"{data:.4f}" , fontdict=dict(fontsize=12), color='white')
    plt.ylim(0.5,1.02)
    plt.savefig(f'{args.dataset}-metrics.png')

graph_bar(f'{args.dataset} predictions - metrics', [accuracy, precision, recall])

# def fairnessMetrics(y_test, y_pred):
#     cm = confusion_matrix(y_test, y_pred)
#     tn, fp, fn, tp = cm.ravel()

#     tpr = tp/(tp+fn)
#     fpr = fp/(tn+fp)

#     #Equalized odds, TPR and FPR
#     print(tpr, fpr)
#     equalized_odds = tpr / fpr

#     return equalized_odds

# equalized_odds = fairnessMetrics(y_test, y_pred)

# logger.info(f'Equalized Odds: {equalized_odds:.4f}')