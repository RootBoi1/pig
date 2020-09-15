from sklearn.feature_selection import (RFECV, VarianceThreshold,
                                       chi2, SelectKBest, SelectFromModel)
from sklearn.linear_model import Lasso
from skrebate import ReliefF
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#####################
# Feature selection methods.
#####################

def svcl1(X_data, y_data, df, args):
    """
    Linear Support Vector Classification with L1 Regularization.
    """
    randseed, C = args
    clf = LinearSVC(penalty="l1", dual=False, random_state=randseed, C=C)
    clf.fit(X_data, y_data)
    print(np.count_nonzero(clf.coef_))
    return [b for a, b in zip(clf.coef_[0], df.columns) if a]

def svcl2(X_data, y_data, df, args):
    """
    Linear Support Vector Classification with L2 Regularization.
    """
    randseed, C = args
    clf = LinearSVC(penalty="l2", random_state=randseed, C=C)
    clf.fit(X_data, y_data)
    sel = SelectFromModel(clf, prefit=True)
    support = sel.get_support(True)
    print(len(support))
    return [b for a, b in zip(clf.coef_[0][support],  df.columns[support])]


def lasso(X_data, y_data, df, alpha=.06):
    mod = Lasso(alpha=alpha)
    mod.fit(X_data, y_data)
    return [b for a, b in zip(mod.coef_, df.columns) if a]


def relief(X_data, y_data, df, param):
    reli = ReliefF()
    reli.fit(X_data, y_data)
    # https://github.com/EpistasisLab/scikit-rebate
    return [df.columns[top] for top in reli.top_features_[:param]]


def variance_threshold(X_data, y_data, df, threshold=0.0):
    clf = VarianceThreshold(threshold)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a]


def select_k_best(X_data, y_data, df, k=20):
    score_func=chi2

    clf = SelectKBest(score_func, k=k)
    mini = 0
    for x in range(0, len(X_data)):
        mini = min(min(X_data[x]), mini)
    if mini < 0:
        for x in range(0, len(X_data)):
            for y in range(0, len(X_data[x])):
                X_data[x][y] -= mini
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a]


def rfecv(X_data, y_data, df, step=1, cv=3):
    rfecv_estimator = SVC(kernel="linear")

    clf = RFECV(rfecv_estimator, step=step, min_features_to_select=20, cv=cv)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a]


def random_forest(X_data, y_data, df, args):
    randseed, max_features = args
    clf = RandomForestClassifier(max_features=max_features, random_state=randseed, n_jobs=1)
    clf.fit(X_data, y_data)
    sel = SelectFromModel(clf, max_features=max_features, prefit=True)
    support = sel.get_support(True)
    print(len(support))
    return [b for a, b in zip(clf.feature_importances_[support],  df.columns[support])]

#######################
# The actual feature selection code.
#######################


def maketasks(folds, df, selection_methods, randseed, debug):
    """Creates the feature selection tasks"""

    tasks = []
    foldnr = 0
    for X_train, X_test, y_train, y_test in folds:
        FOLDXY = (X_train, X_test, y_train, y_test)
        # Each task: (i, type, FOLDXY, df, (args)
        if debug:
            tasks.extend([(foldnr, "Lasso", FOLDXY, df, .05),
                          (foldnr, "Relief", FOLDXY, df, 40),
                          (foldnr, "VarThresh", FOLDXY, df, 1)])

        else:
            for method, parameters in selection_methods.items():
                if method == 'Lasso':
                    for alpha in parameters: # [.05, 0.1]
                        tasks.append((foldnr, "Lasso", FOLDXY, df, alpha))
                if method == 'VarThresh':
                    for threshold in parameters: # [.99, .995, 1, 1.005, 1.01]
                        tasks.append((foldnr, "VarThresh", FOLDXY, df, threshold))
                if method == 'SelKBest':
                    for k in parameters: # [20]
                        tasks.append((foldnr, "SelKBest", FOLDXY, df, k))
                if method == 'Relief':
                    for features in parameters: # [40, 60, 80]
                        tasks.append((foldnr, "Relief", FOLDXY, df, features))          
                if method == 'RFECV':
                    for stepsize in parameters: # [1, 2, 3]
                        tasks.append((foldnr, "RFECV", FOLDXY, df, stepsize))
                if method == 'SVC1':
                    for C in parameters:
                        tasks.append((foldnr, "SVC1", FOLDXY, df, (randseed, C)))
                if method == 'SVC2':
                    for C in parameters:
                        tasks.append((foldnr, "SVC2", FOLDXY, df, (randseed, C)))
                if method == 'Forest':
                    for max_features in parameters:
                        tasks.append((foldnr, "Forest", FOLDXY, df, (randseed, max_features)))
        foldnr += 1

                    
      
    np.array(tasks, dtype=object).dump("tmp/fs_tasks")
    return tasks


def feature_selection(taskid):
    """Executes the feature selection using the given task.
    Args:
      taskid: An ID for a made from maketasks()

    Returns:
      featurelist(List)
      """
    tasks = np.load("tmp/fs_tasks", allow_pickle=True)
    foldnr, fstype, FOLDXY, df, args = tasks[taskid]
    X_train, X_test, y_train, y_test = FOLDXY
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
    else:
        raise ValueError(f"'{fstype}' is not a valid Feature selection method.")
    mask = [True if f in fl else False for f in df.columns]
    return foldnr, fl, mask, f"{fstype}: {args}", FOLDXY
