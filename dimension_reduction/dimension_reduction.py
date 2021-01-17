import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


def dimension_reduction(foldxy, mask, dr, seed):
    X_train, X_test, y_train, y_test = foldxy
    X_train = StandardScaler().fit_transform(X_train)
    X_train = np.array(X_train)[:,mask]
    X_test = np.array(X_test)[:,mask]
    model = None
    if dr == "pca":
        model =  make_pipeline(StandardScaler(), PCA(random_state=seed))
        model.fit(X_train, y_train)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
    elif dr == "autoencoder":
        pass
    elif dr == "umap":
        pass
    elif dr == "nca":
        model = make_pipeline(StandardScaler(),
                NeighborhoodComponentsAnalysis(random_state=seed)) 
        model.fit(X_train, y_train)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
    foldxy = []
    foldxy.append(X_train)
    foldxy.append(X_test)
    foldxy.append(y_train)
    foldxy.append(y_test)
    return foldxy, model
