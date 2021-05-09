import numpy as np
import umap
import dimension_reduction.autoencoder as ae
import torch
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
    if dtype == "autoencoder":
        # Autoencoder
        vae = ae.VAE(num_features, int((X_train.shape[1]+num_features)/2), X_train.shape[1])
        tensor_X_train = torch.from_numpy(X_train).float().to(torch.device('cpu'))
        data_loader = torch.utils.data.DataLoader(tensor_X_train, batch_size=300, shuffle=False, num_workers=1)
        # Train the autoencoder
        vae.train(data_loader, epochs=50, beta=0.00)
        # Encode X_train
        X_train = vae.predict(data_loader).detach().numpy()
        # Autoencode X_test
        tensor_X_test = torch.from_numpy(X_test).float().to(torch.device('cpu'))
        data_loader = torch.utils.data.DataLoader(tensor_X_test, batch_size=300, shuffle=False, num_workers=1)
        X_test = vae.predict(data_loader).detach().numpy()
        foldxy[0] = X_train
        foldxy[1] = X_test
        return foldxy, True
    model = None
    if dtype == "pca":
        model = make_pipeline(StandardScaler(),
                PCA(n_components=num_features, random_state=seed))
    elif dtype == "umap":
        model = make_pipeline(StandardScaler(),
                umap.UMAP(n_components=num_features, n_neighbors=200, n_epochs=500, random_state=seed))
    elif dtype == "lda":
        model = LinearDiscriminantAnalysis(n_components=num_features)
    elif dtype == "nca":
        model = make_pipeline(StandardScaler(),
                NeighborhoodComponentsAnalysis(random_state=seed, n_components=num_features)) 
    model.fit(X_train, y_train)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    foldxy[0] = X_train
    foldxy[1] = X_test
    return foldxy, model
