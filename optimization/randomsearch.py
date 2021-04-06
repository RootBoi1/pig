# define parameters for the random search 

import pprint
from skopt.space import Real, Categorical, Integer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from scipy.stats import randint as rint
from imblearn.over_sampling import RandomOverSampler

class OS_MLPClassifier(MLPClassifier):
    def fit(self, X, y):
        os = RandomOverSampler(random_state=42)
        X_re, y_re = os.fit_sample(X, y)
        MLPClassifier.fit(self, X_re, y_re)

class OS_ExtraTreesClassifier(ExtraTreesClassifier):
    def fit(self, X, y):
        os = RandomOverSampler(random_state=42)
        X_re, y_re = os.fit_sample(X, y)
        ExtraTreesClassifier.fit(self, X_re, y_re)

class OS_GradientBoostingClassifier(GradientBoostingClassifier):
    def fit(self, X, y):
        os = RandomOverSampler(random_state=42)
        X_re, y_re = os.fit_sample(X, y)
        GradientBoostingClassifier.fit(self, X_re, y_re)

neuralnet_params = {'activation': ['relu','tanh','logistic','identity'], # MLP
     'alpha': [0.0001,0.001,0.01,0.00001],
     'batch_size': ['auto'],
     'beta_1': [0.9],
     'beta_2': [0.999],
     'early_stopping': [False],
     'epsilon': [1e-08],
     'hidden_layer_sizes': [(x,y)for x in range(50,251) for y in range(1,6)],
     'learning_rate': ['constant','adaptive'],
     'learning_rate_init': [0.001],
     'max_iter': list(range(500,2000)),
     'momentum': [0.9],
     'n_iter_no_change': [10],
     'nesterovs_momentum': [True],
     'power_t': [0.5],
     'random_state': [None],
     'shuffle': [True],
     'solver': ['adam'],
     'tol': [0.0001],
     'validation_fraction': [0.1],
     'verbose': [False],
     'warm_start': [False]}
xtratrees_params = {'bootstrap': [False,True], # xtra trees
     'class_weight': ['balanced'],
     'criterion': ['gini','entropy'],
     'max_depth': [None],
     'max_features': ['auto'],
     'max_leaf_nodes': [None],
     'min_impurity_decrease': [0.0],
     'min_impurity_split': [None],
     'min_samples_leaf': [1],
     'min_samples_split': [2],
     'min_weight_fraction_leaf': [0.0],
     'n_jobs': [None],
     'n_estimators': list(range(20,300)),
     'oob_score': [False],
     'random_state': [None],
     'verbose': [0],
     'warm_start': [False]}
gradientboosting_params = {'criterion': ['friedman_mse'],  # gradient boosting
     'init': [None],
     'learning_rate': [0.1, 0.2, 0.05, 0.3],
     'loss': ['deviance','exponential'],
     'max_depth': [5,7,10,13,16],
     'max_features': ['sqrt','log2', None],
     'max_leaf_nodes': [None],
     'min_impurity_decrease': [0.0],
     'min_impurity_split': [None],
     'min_samples_leaf': [1],
     'min_samples_split': [2, 3, 4],
     'min_weight_fraction_leaf': [0.0],
     'n_estimators':  list(range(50, 400)),
     'n_iter_no_change': [None],
     'random_state': [None],
     'subsample': [1.0],
     'tol': [0.0001, 0.001, 0.00001],
     'validation_fraction': [0.1],
     'verbose': [0],
     'warm_start': [False]}

gradientboosting_skopt_params = {'learning_rate': Real(0.001, 0.9),
     'loss': Categorical(['deviance','exponential']),
     'max_depth': Integer(5,16),
     'max_features': Categorical(['sqrt','log2', None]),
     'min_samples_leaf': Integer(1, 7),
     'min_samples_split': Integer(2, 7),
     'n_estimators':  Integer(50, 400),
     'tol': Real(0.00001, 0.1)}


classifiers = { # {clfname: (Classifier, {parameters}), ...}
    'neuralnet': (MLPClassifier(), neuralnet_params), 
    'xtratrees': (ExtraTreesClassifier(), xtratrees_params), 
    'gradientboosting': (GradientBoostingClassifier(), gradientboosting_skopt_params),
    'os_neuralnet': (OS_MLPClassifier(), neuralnet_params),
    'os_xtratrees': (OS_ExtraTreesClassifier(), xtratrees_params),
    'os_gradientboosting': (OS_GradientBoostingClassifier(), gradientboosting_params)
    }


##classifiers = [
##    #KNeighborsClassifier(),
##    #SVC(), 2 slow 
##    #DecisionTreeClassifier(),
##    MLPClassifier(), #
##    #AdaBoostClassifier(),
##    #GaussianNB(),
##    #QuadraticDiscriminantAnalysis(),
##    #RandomForestClassifier(),
##    ExtraTreesClassifier(),
##    GradientBoostingClassifier()
##    ]
##
##param_lists = [
###  {'algorithm': ['auto'], # kneighs
###  'leaf_size': [30], 
###  'metric': ['minkowski'],
###  'p': [.5,1,1.5,2,2.5,3],
###  'n_jobs': [None],
###  'n_neighbors': [1,2,3,4,5,6,7,8,9,10],
###  'weights': ['uniform','distance']},
### {'C': [10**x for x in range(-2,6)], # svc 
###  'cache_size': [200],
###  'class_weight': ['balanced'],
###  'coef0': [0.0],
###  'decision_function_shape': ['ovr'],
###  'degree': [2,3,4,5],
###  'gamma': ['auto_deprecated' ],
###  'kernel': ["linear", "poly", "rbf", "sigmoid" ],
###  'max_iter': [-1],
###  'probability': [False],
###  'random_state': [None],
###  'shrinking': [True],
###  'tol': [0.001],
###  'verbose': [False]},
### {'class_weight': [None], # decision tree classifier 
###  'criterion': ['gini','entropy'],
###  'max_depth': [None],
###  'max_features': [None,'sqrt','log2'],
###  'max_leaf_nodes': [None],
###  'min_impurity_decrease': [0.0],
###  'min_impurity_split': [None],
###  'min_samples_leaf': [1],
###  'min_samples_split': [2],
###  'min_weight_fraction_leaf': [0.0],
###  'presort': [False],
###  'random_state': [None],
###  'splitter': ['best']},
## {'activation': ['relu','tanh','logistic','identity'], # MLP 
##  'alpha': [0.0001,0.001,0.01,0.00001],
##  'batch_size': ['auto'],
##  'beta_1': [0.9],
##  'beta_2': [0.999],
##  'early_stopping': [False],
##  'epsilon': [1e-08],
##  'hidden_layer_sizes': [(x,)for x in range(20,500)],
##  'learning_rate': ['constant','adaptive'],
##  'learning_rate_init': [0.001],
##  'max_iter': list(range(200,2000)),
##  'momentum': [0.9],
##  'n_iter_no_change': [10],
##  'nesterovs_momentum': [True],
##  'power_t': [0.5],
##  'random_state': [None],
##  'shuffle': [True],
##  'solver': ['adam'],
##  'tol': [0.0001],
##  'validation_fraction': [0.1],
##  'verbose': [False],
##  'warm_start': [False]},
###{'algorithm': ['SAMME.R'],# adaboost 
### 'base_estimator': [None],
### 'learning_rate': [1.0],
### 'n_estimators': [50],
### 'random_state': [None]},
###{'priors': [None], 'var_smoothing': [1e-09]}, # bla
###{'priors': [None], # bla
### 'reg_param': [0.0],
### 'store_covariance': [False],
### 'tol': [0.0001]},
### {'bootstrap': [True,False],
###  'class_weight': ['balanced'],
###  'criterion': ['gini','entropy'],
###  'max_depth': [None],
###  'max_features': [None,'sqrt','log2'],
###  'max_leaf_nodes': [None],
###  'min_impurity_decrease': [0.0],
###  'min_impurity_split': [None],
###  'min_samples_leaf': [1],
###  'min_samples_split': [2],
###  'min_weight_fraction_leaf': [0.0],
###  'n_estimators': [20,50,100,150],
###  'n_jobs': [None],
###  'oob_score': [False,True],
###  'random_state': [None],
###  'verbose': [0],
###  'warm_start': [False]},
## {'bootstrap': [False,True], # xtra trees 
##  'class_weight': ['balanced'],
##  'criterion': ['gini','entropy'],
##  'max_depth': [None],
##  'max_features': ['auto'],
##  'max_leaf_nodes': [None],
##  'min_impurity_decrease': [0.0],
##  'min_impurity_split': [None],
##  'min_samples_leaf': [1],
##  'min_samples_split': [2],
##  'min_weight_fraction_leaf': [0.0],
##  'n_jobs': [None],
##  'n_estimators': list(range(20,300)),
##  'oob_score': [False],
##  'random_state': [None],
##  'verbose': [0],
##  'warm_start': [False]},
## {'criterion': ['friedman_mse'], # gradient boosting 
##  'init': [None],
##  'learning_rate': [0.1,0.001,0.0001,0.3],
##  'loss': ['deviance','exponential'],
##  'max_depth': [3,5,7],
##  'max_features': [None,'sqrt','log2'],
##  'max_leaf_nodes': [None],
##  'min_impurity_decrease': [0.0],
##  'min_impurity_split': [None],
##  'min_samples_leaf': [1],
##  'min_samples_split': [2],
##  'min_weight_fraction_leaf': [0.0],
##  'n_estimators':  list(range(20,300)),
##  'n_iter_no_change': [None],
###  'presort': ['auto'],
##  'random_state': [None],
##  'subsample': [1.0],
##  'tol': [0.0001],
##  'validation_fraction': [0.1],
##  'verbose': [0],
##  'warm_start': [False]}]
##
##
##if __name__=="__main__":
##    # get initial stuff, automate as much has possible
##    valtolist = lambda x: { k:[v] for k,v in x.items()}
##    param_list = [ valtolist(clf.get_params()) for clf in classifiers]
##    pprint.pprint(param_list)
##
