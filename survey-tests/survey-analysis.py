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


#print(dataset['race'].value_counts().sort_values())

qtd_train = (len(dataset) * args.percentage) // 100

x = dataset.drop([args.label], axis=1)
x_train = x[:qtd_train]
x_test = x[qtd_train:]
y = dataset[args.label]
y_train = y[:qtd_train]
y_test = y[qtd_train:]

logger.info(f'\nRunning training with {args.percentage}% of data.')
logger.info(f'Train Lines: {len(x_train)}')
logger.info(f'Test Lines: {len(x_test)}')


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=60000)

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,plot_confusion_matrix

logger.info(u'\nMetric Values of Logistic Regression Predictions:')
logger.info(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
logger.info(f'Precision: {precision_score(y_test, y_pred):.4f}')
logger.info(f'Recall: {recall_score(y_test, y_pred):.4f}')