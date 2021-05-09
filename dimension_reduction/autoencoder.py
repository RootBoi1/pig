import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np


class EncoderFC(torch.nn.Module):
    def __init__(self, input_dim, hidden, latent_dim):
        super(EncoderFC, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, int((hidden+input_dim)/2))
        self.linear2 = torch.nn.Linear(int((input_dim+hidden)/2), hidden)
        self.linear41 = torch.nn.Linear(hidden, latent_dim)
        self.linear42 = torch.nn.Linear(hidden, latent_dim)
  
    def forward(self, x): 
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear41(x), F.softplus(-self.linear42(x))

class DecoderFC(torch.nn.Module):
    def __init__(self, latent_dim, hidden, input_dim):
        super(DecoderFC, self).__init__()
        self.linear1 = torch.nn.Linear(latent_dim, hidden)
        self.linear2 = torch.nn.Linear(hidden, int((input_dim+hidden)/2))
        self.linear4 = torch.nn.Linear(int((input_dim+hidden)/2), input_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return torch.sigmoid(self.linear4(x))

class VAE(torch.nn.Module):
    def __init__(self, latent_dim, first_layer_size, input_size):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim 
        self.encoder = EncoderFC(input_size, first_layer_size, self.latent_dim)
        self.decoder = DecoderFC(self.latent_dim, first_layer_size, input_size)
        self.device = 'cpu'
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-03)
       
    def reparamterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparamterize(mu, logvar)
        return z, mu, logvar

    def decode(self, data):
        pred = self.decoder(data)
        return pred

    def forward(self, data):
        # Encode the input data to obtain latent vector z and latent distribution mu/logvar
        z, mu, logvar = self.encode(data)
        # Decode the latent representation to obtain the predicte image
        decoded = self.decode(z)
        return decoded, z, mu, logvar
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def lossfunction(self, data, pred, beta, mu, logvar):
        # Reconstruction loss
        rec = F.mse_loss(pred, data, reduction='mean')
        # KLD to unit gaussian
        kld = (-beta * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean(0,True)
        return rec + kld

    def train(self, dataset, epochs, beta):
        # Iterate multiple times through the entire dataset
        for e in range(epochs):
            losses = []
            # Iterate through dataset (1 time through all batches in the dataset)
            for i, data in enumerate(dataset, 0):
                # Get a batch from the data and put it on the CPU
                img = data.to(self.device).float()
                # Forward pass the input through the VAE
                pred, z, mu, logvar = self(img)
                loss = self.lossfunction(img, pred, beta, mu, logvar)
                # Step the optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # housekeeping
                losses.append(loss.item())
            print("Epoch {}/{} done, loss: {}!".format(e+1, epochs, np.mean(losses)))


    def predict(self, dataset):
        for i, data in enumerate(dataset, 0):
            # Get a batch from the data and put it on the CPU
            img = data.to(self.device).float()
            # Forward pass the input through the VAE
            pred, z, mu, logvar = self(img)
            # Accumulate latentspace
            if i == 0:
                z_full = z
            else:
                z_full = torch.cat((z_full, z), 0)
            if i == len(dataset) - 1:
                return z_full


# Hyperparameter
# - beta: How similar the latent distribution given my mu and logvar has to be to a gaussian unit distribution
# - learning rate (see optimizer)
# - batch size: how many images per forward 
# - Network architecture
