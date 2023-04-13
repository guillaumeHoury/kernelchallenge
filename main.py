from algos.kernelrr import KernelRR

from kernels.feature_vector_kernels import DegreeHistogramKernel, EdgeLabelHistogramKernel
from kernels.graph_kernels import WLKernel
from kernels.combine_kernels import SumKernel

import pickle as pkl
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold

import warnings
import time

from sklearn import metrics


def load_data():
    with open('data/training_data.pkl', 'rb') as file:
        train_graphs = np.array(pkl.load(file))
    with open('data/test_data.pkl', 'rb') as file:
        test_graphs = np.array(pkl.load(file))
    with open('data/training_labels.pkl', 'rb') as file:
        train_labels = np.array(pkl.load(file))

    return train_graphs, test_graphs, train_labels


def test_classifier(graphs, labels, classifier):
    """
    Test the classifier on a test split of the dataset.
    """

    train_graphs, test_graphs, train_labels, test_labels = train_test_split(graphs, labels, test_size=0.3, random_state=None)

    classifier.fit(train_graphs, train_labels)

    train_preds = classifier.predict(train_graphs)
    test_preds = classifier.predict(test_graphs)

    print("Train Accuracy {}".format(metrics.roc_auc_score(train_labels, train_preds)))
    print("Test Accuracy {}".format(metrics.roc_auc_score(test_labels, test_preds)))


def kfold_test(graph, labels, classifier, n_splits=3):
    """
    Test the classifier on the whole dataset using kfold cross-validation.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=20)
    test_score = 0.

    for i, (train_index, test_index) in enumerate(kf.split(graph)):
        train_graphs, train_labels = graph[train_index], labels[train_index]
        test_graphs, test_labels = graph[test_index], labels[test_index]

        classifier.fit(train_graphs, train_labels)
        test_preds = classifier.predict(test_graphs)
        cur_test_score = metrics.roc_auc_score(test_labels, test_preds)
        test_score += cur_test_score
        print("Fold {}, test Accuracy {}".format(i, cur_test_score))
    print("Total, test Accuracy {}".format(test_score/n_splits))


def compute_predictions(train_graphs, test_graphs, train_labels, classifier):
    """
    Compute the predictions on the test set and store it in a file 'test_pred.csv'
    """
    classifier.fit(train_graphs, train_labels)

    test_preds = classifier.predict(test_graphs)

    Yte = {'Predicted': test_preds}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv('test_pred.csv', index_label='Id')


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    train_graphs, test_graphs, train_labels = load_data()
    kernel = WLKernel(edge_attr=True, node_attr=True, iterations=10).kernel
    classifier = KernelRR(lmbda=5e-5, kernel=kernel, verbose=False)

    # kfold_test(train_graphs, train_labels, classifier, n_splits=3)
    compute_predictions(train_graphs, test_graphs, train_labels, classifier)
