from sklearn.feature_selection import (RFECV, VarianceThreshold,
                                       chi2, SelectKBest, SelectFromModel)
from sklearn.linear_model import Lasso
from skrebate import ReliefF, SURF
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
#####################
# Feature selection methods.
#####################

def svcl1(X_data, y_data, df, args):
    """
    Linear Support Vector Classification with L1 Regularization.
    """
    randseed, C = args
    clf = LinearSVC(penalty="l1", dual=False, random_state=randseed,
                    C=C)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.coef_[0], df.columns) if a]

def svcl2(X_data, y_data, df, args):
    """
    Linear Support Vector Classification with L2 Regularization.
    """
    randseed, max_features = args
    clf = LinearSVC(penalty="l2", random_state=randseed)
    clf.fit(X_data, y_data)
    max_features = int(max_features)
    sel = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=max_features)
    support = sel.get_support(True)
    return [b for a, b in zip(clf.coef_[0][support],  df.columns[support])]


def lasso(X_data, y_data, df, alpha):
    clf = Lasso(alpha=alpha)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.coef_, df.columns) if a]


def relief(X_data, y_data, df, param):
    clf = ReliefF(n_neighbors=50)
    clf.fit(X_data, y_data)
    # https://github.com/EpistasisLab/scikit-rebate
    return [df.columns[top] for top in clf.top_features_[:param]]


def variance_threshold(X_data, y_data, df, threshold):
    clf = VarianceThreshold(threshold)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a]


def select_k_best(X_data, y_data, df, k):
    score_func=chi2

    clf = SelectKBest(score_func, k=k)
    """mini = 0
    for x in range(0, len(X_data)):
        mini = min(min(X_data[x]), mini)
    if mini < 0:
        for x in range(0, len(X_data)):
            for y in range(0, len(X_data[x])):
                X_data[x][y] -= mini"""
    if X_data.min() < 0:
        X_data += abs(X_data.min())
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a]


def rfecv(X_data, y_data, df, step, cv):
    rfecv_estimator = SVC(kernel="linear")

    clf = RFECV(rfecv_estimator, step=step, min_features_to_select=20, cv=cv)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a]


def random_forest(X_data, y_data, df, args):
    randseed, threshold= args
    clf = RandomForestClassifier(random_state=randseed, n_jobs=1)
    clf.fit(X_data, y_data)
    sel = SelectFromModel(clf, threshold=threshold, prefit=True)
    support = sel.get_support(True)
    return [b for a, b in zip(clf.feature_importances_[support],  df.columns[support])]


def agglomerative_clustering(X_data, y_data, df, args):
    seed, n_clusters = args
    clf = AgglomerativeClustering(n_clusters=n_clusters) 
    X_data = np.transpose(X_data)
    clf.fit(X_data)
    labels = clf.labels_
    fl = []
    for i in range(n_clusters):
        indices = [x for x, y in enumerate(labels) if y==i]
        dataframe = pd.DataFrame(X_data).transpose().iloc[:, indices]
        dataframe.insert(0, "labels", y_data, allow_duplicates=True)
        corr_matrix = dataframe.corr("spearman").abs().iloc[:,[0]]
        best_feature_index = np.argmax(corr_matrix.values[1:])
        fl.append(list(df.columns)[indices[best_feature_index]])
        #dataf = pd.DataFrame(X_data).transpose().iloc[:, indices]
        #vt = VarianceThreshold()
        #vt.fit(dataf)
        #fl.append(list(df.columns)[indices[list(vt.variances_).index(max(vt.variances_))]])
    return fl


def random(X_data, y_data, df, args):
    """
    Args:
      args: (num_features, seed)
    Make sure 1, 20001 40001 etc have same seed.
    """
    num_features, seed = args

    np.random.seed(seed)
    return np.random.choice(df.columns, num_features, replace=False).tolist()

#######################
# The actual feature selection code.
#######################


def feature_selection(foldxy, fstype, args, df):
    """Executes the feature selection using the given task.
    Args:
      task: A FS task made by maketasks() above
      foldxy: [X_train, X_test, y_train, y_test]
      df: The used dataframe

    Returns:
      featurelist(List)
      """
    X_train, X_test, y_train, y_test = foldxy
    if fstype == "Lasso":
        fl = lasso(X_train, y_train, df, args)
    elif fstype == "Relief":
        fl = relief(X_train, y_train, df, args)
    elif fstype == "VarThresh":
        fl = variance_threshold(X_train, y_train, df, args)
    elif fstype == "SelKBest":
        fl = select_k_best(X_train, y_train, df, args)
    elif fstype == "RFECV":
        fl = rfecv(X_train, y_train, df, args)
    elif fstype == "SVC1":
        fl = svcl1(X_train, y_train, df, args)
    elif fstype == "SVC2":
        fl = svcl2(X_train, y_train, df, args)
    elif fstype == "Forest":
        fl = random_forest(X_train, y_train, df, args)
    elif fstype == "Random":
        fl = random(X_train, y_train, df, args)
    elif fstype == "AggloClust":
        fl = agglomerative_clustering(X_train, y_train, df, args)
    else:
        raise ValueError(f"'{fstype}' is not a valid Feature selection method.")
    mask = [True if f in fl else False for f in df.columns]
    return fl, mask, f"{fstype}: {args}"
