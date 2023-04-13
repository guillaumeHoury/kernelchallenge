from algos.kernelrr import KernelRR

from kernels.feature_vector_kernels import DegreeHistogramKernel, EdgeLabelHistogramKernel
from kernels.graph_kernels import NthWalkKernel, WLKernel
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

def sub_index(train_graphs, train_labels, N_Zeros, N_Ones):
    # selection d'une sous partie des données
    train_index = np.arange(len(train_graphs))
    one_index = train_index[train_labels == 1]
    zero_index = train_index[train_labels == 0]
    np.random.seed(0)
    sub_zero_index = np.random.choice(zero_index, size = N_Zeros, replace= None)
    sub_one_index = np.random.choice(one_index, size = N_Ones, replace= None)
    sub_train_index = np.concatenate((sub_one_index , sub_zero_index))
    return train_graphs[sub_train_index], train_labels[sub_train_index]

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    train_graphs, test_graphs, train_labels = load_data()
    train_graphs, train_labels =  sub_index(train_graphs, train_labels, N_Zeros = 1000, N_Ones = 500)
    
    list_param = [5]
    for param in list_param:
        print(f"param: {param} ")
        kernel = NthWalkKernel(walk_length=param).kernel
        classifier = KernelRR(lmbda=5e-5, kernel=kernel, verbose=False)
        kfold_test(train_graphs, train_labels, classifier, n_splits=3)
""""
    list_alphas = [[0.7, 0.1, 1.0]]#, [0.6,.2]]#, [0.7, .1, 1.], [0.7, .5, 0.5], [0.8, .6, 0.5], [0.7, .2, 0.5], [.1, .1,  1.]]
    for alphas in list_alphas:
        print(f"alphas: {alphas} ")
        kernel = SumKernel([EdgeLabelHistogramKernel().kernel,
                                DegreeHistogramKernel(max_degree=4).kernel,
                                WLKernel(edge_attr=True, node_attr=True, iterations=10).kernel],
                                alphas=alphas).kernel
        classifier = KernelRR(lmbda=5e-5, kernel=kernel, verbose=False)
        

        #kfold_test(train_graphs, train_labels, classifier, n_splits=3)
        compute_predictions(train_graphs, test_graphs, train_labels, classifier)"""
