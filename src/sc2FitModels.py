""" Score a feature matrix on either tremor, dyskinesia, or bradykinesia tasks.

See function `score` if calling from within Python.
Pass the -h flag if calling from the command line.

Author: Phil Snyder (phil [dot] snyder [at] sagebase [dot] org)
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import synapseclient as sc
import argparse
from nonLinearInterpAUPRC import getAUROC_PR
from sklearn.metrics.base import _average_binary_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

TRAINING_TABLE = "syn10495809"
INDEX_COL = "dataFileHandleId"
CATEGORY_WEIGHTS = {
        'tremor':[896, 381, 214, 9, 0],
        'dyskinesia':[531, 129], # no weighting is done when
        'bradykinesia':[990, 419]} # scoring the binary cases

syn = sc.login()
np.random.seed(0) # reproducible results for stochastic training elements

def read_args():
    """ Relevant only if calling from command line. """
    parser = argparse.ArgumentParser(
            description="Score an L-Dopa Submission file (AUPRC).")
    parser.add_argument("phenotype",
            help="One of 'tremor', 'dyskinesia', or 'bradykinesia'")
    parser.add_argument("submission",
            help="filepath to submissions file")
    args = parser.parse_args()
    return args

def read_data(path, phenotype):
    """ Read in training data from `path` and fetch targets for `phenotype`.

    Parameters
    ----------
    path (str) - filepath to submission feature matrix
    phenotype (str) - one of 'tremor', 'dyskinesia', 'bradykinesia'

    Returns
    -------
    train_X, train_y (numpy.ndarray)
    """
    df = pd.read_csv(path, index_col=INDEX_COL, header=0)
    train_table = get_table(TRAINING_TABLE)
    train_ids = train_table.index.values
    train = df.loc[train_ids]
    train = train.join(
            train_table[['tremorScore', 'dyskinesiaScore', 'bradykinesiaScore']])
    train_y = train["{}Score".format(phenotype)]
    train = train.drop(['tremorScore', 'dyskinesiaScore', 'bradykinesiaScore'],
            axis=1)
    train_not_na = [pd.notnull(v) for v in train_y]
    train_X = train[train_not_na].values
    train_y = train_y[train_not_na].values
    return train_X, train_y

def get_table(synId):
    """ Returns all rows from a Synapse Table as a DataFrame """
    q = syn.tableQuery("select * from {}".format(synId))
    df = q.asDataFrame()
    df = df.set_index(INDEX_COL, drop=True)
    return df

def train_ensemble(X, y):
    """ Trains a soft-voting ensemble consisting of a random forest,
    logistic regression with L2 regularization, and support vector
    machine (RBF kernel).

    Parameters
    ----------
    X (numpy.ndarray) - feature matrix
    y (numpy.ndarray) - target array

    Returns
    -------
    ensemble (VotingClassifier)
    """
    rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=500))
    lr = OneVsRestClassifier(LogisticRegressionCV())
    svm = OneVsRestClassifier(SVC(probability=True))
    ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr), ('svm', svm)], voting='soft')
    ensemble.fit(X, y)
    return ensemble

def getNonLinearInterpAupr(X, y, y_labels, clf):
    """ Given data, targets, a list of unique target values, and an
    sklearn classifier, return the weighted non-linear interpolated AUPRC.

    Parameters
    ----------
    X (np.ndarray) - feature matrix
    y (np.ndarray) - target values
    y_labels (np.ndarray) - unique list of target values (in case missing target
        values in test set)
    clf (sklearn.base.BaseEstimator) - an sklearn object with method `predict_proba`
    average (str, default 'micro') - one of 'micro', 'macro', 'samples', 'weighted'
        (see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
    """
    n_classes = len(y_labels)
    if n_classes > 2:
        y_true = label_binarize(y, y_labels)
    else:
        y_true = y
    y_score = clf.predict_proba(X) if n_classes > 2 else clf.predict_proba(X).T[1]
    return nonLinearInterpAupr(y_score, y_true)

def nonLinearInterpAupr(y_score, y_true):
    """ Given ground truth targets and predicted targets, calculates
    weighted non-linear interpolated AUPRC.

    Parameters
    ----------
    y_true (list-like) - actual target values
    y_score (list-like) - predicted target values
    average (str, default 'micro') - one of 'micro', 'macro', 'samples', 'weighted'
        (see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
    sample_weight (list-like, optional) - of shape (n_samples,)

    Returns
    -------
    results (list), y_score (np.ndarray), y_true (np.ndarray)
    """
    if y_score.ndim > 1:
        sub_stats = pd.DataFrame(np.concatenate((y_score, y_true), axis=1),
                dtype='float64')
    else:
        sub_stats = pd.DataFrame(np.array([y_score, y_true]).T)
    results = getAUROC_PR(sub_stats)
    return results, y_score, y_true

def getWeightedMean(phenotype, scores):
    numer = 0
    denom = sum(CATEGORY_WEIGHTS[phenotype])
    for w, s in zip(CATEGORY_WEIGHTS[phenotype], scores):
        if pd.notnull(s):
            numer += w * s
    return numer / denom

def calculatePVal(y_score, y_true, trueAupr, phenotype):
    """ Calculate P-value for a set of predictions.

    Parameters
    ---------
    y_score (np.ndarray) - predicted targets
    y_true (np.ndarray) - actual targets
    trueAupr (float, optional) - actual AUPRC
    phenotype (str) - one of 'tremor', 'dyskinesia', 'bradykinesia'

    Returns
    -------
    pval (float)
    """
    auprs = []
    n_iterations = 10
    for i in range(n_iterations):
        np.random.shuffle(y_score)
        results = nonLinearInterpAupr(y_score, y_true)
        if phenotype == 'tremor':
            weighted_results = getWeightedMean(phenotype, results)
        else:
            weighted_results = results[0]
        auprs.append(weighted_results)
    return sum([a > trueAupr for a in auprs]) / n_iterations

def score(phenotype, submission):
    """ Returns AUPRC, predicted and actual targets on
    training data given a feature matrix.

    Parameters
    ----------
    phenotype (str) - one of 'tremor', 'dyskinesia', 'bradykinesia'
    submission (str) - filepath to submissions file

    Returns
    -------
    aupr (float)
    pval (float)
    y_score (np.ndarray)
    y_true (np.ndarray)
    """
    train_X, train_y = read_data(submission, phenotype)
    ensemble = train_ensemble(train_X, train_y)
    results, y_score, y_true = getNonLinearInterpAupr(train_X, train_y,
            np.arange(len(CATEGORY_WEIGHTS[phenotype])), ensemble)
    if phenotype == 'tremor':
        weighted_aupr = getWeightedMean(phenotype, results)
    else:
        weighted_aupr = results[0]
    # pval calculation is very slow
    #pval = calculatePVal(y_score, y_true, weighted_aupr, phenotype)
    return weighted_aupr, y_score, y_true

def main():
    """ For use on command line. Only returns AUPRC"""
    args = read_args()
    aupr, y_score, y_true = score(args.phenotype, args.submission)
    print("AUPRC: {}".format(aupr))
    return aupr

#if __name__ == "__main__":
#    main()
