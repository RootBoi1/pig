import numpy as np
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


def dimension_reduction(foldxy, mask, dr, seed):
    dtype, num_features = dr
    X_train, X_test, y_train, y_test = foldxy
    X_train = np.array(X_train)[:,mask]
    X_test = np.array(X_test)[:,mask]
    X_train = StandardScaler().fit_transform(X_train)
    model = None
    if dtype == "pca":
        model = make_pipeline(StandardScaler(),
                PCA(n_components=num_features, random_state=seed))
    elif dtype == "autoencoder":
        pass
    elif dtype == "umap":
        model = make_pipeline(StandardScaler(), umap.UMAP(n_components=num_features))
    elif dtype == "lda":
        model = make_pipeline(StandardScaler(),
                LinearDiscriminantAnalysis(n_components=num_features))
    elif dtype == "nca":
        model = make_pipeline(StandardScaler(),
                NeighborhoodComponentsAnalysis(random_state=seed, n_components=num_features)) 
    model.fit(X_train, y_train)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    foldxy[0] = X_train
    foldxy[1] = X_test
    return foldxy, model
