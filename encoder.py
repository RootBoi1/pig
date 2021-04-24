import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np

# opencv to show images
import cv2
def show_image(true, pred):
    img = np.concatenate((np.vstack(true), np.vstack(pred)), axis=1)
    cv2.imshow("vis", img)
    cv2.waitKey(1000)

class EncoderFC(torch.nn.Module):
    def __init__(self, input_dim, hidden, latent_dim):
        super(EncoderFC, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden)
        self.linear2 = torch.nn.Linear(hidden, hidden)
        self.linear3 = torch.nn.Linear(hidden, hidden)
        self.linear41 = torch.nn.Linear(hidden, latent_dim)
        self.linear42 = torch.nn.Linear(hidden, latent_dim)
  
    def forward(self, x): 
        # Flatten tensor input
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear41(x), F.softplus(-self.linear42(x))

class DecoderFC(torch.nn.Module):
    def __init__(self, latent_dim, hidden, input_dim):
        super(DecoderFC, self).__init__()
        self.linear1 = torch.nn.Linear(latent_dim, hidden)
        self.linear2 = torch.nn.Linear(hidden, hidden)
        self.linear3 = torch.nn.Linear(hidden, hidden)
        self.linear4 = torch.nn.Linear(hidden, input_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return torch.sigmoid(self.linear4(x))

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.latent_dim = 8
        self.encoder = EncoderFC(28*28, 300, self.latent_dim)
        self.decoder = DecoderFC(self.latent_dim, 300, 28*28)
        self.device = 'cpu' # cuda if you have a nvidia gpu
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
        rec = F.binary_cross_entropy(pred, data.view(-1, data.shape[1] * data.shape[2]), reduction='mean')
        # KLD to unit gaussian
        kld = (-beta * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean(0,True)
        return rec + kld

    def train(self, dataset, epochs, beta):
        # Iterate multiple times through the entire dataset
        for e in range(epochs):
            losses = []
            # Iterate through dataset (1 time through all batches in the dataset)
            for i, data in enumerate(dataset, 0):
                # Get a batch of images from the data and put it on the GPU (if used)
                img = torch.from_numpy(np.array(data[0])).squeeze(1).to(self.device).float()
                # Forward pass the input image img through the VAE
                pred, z, mu, logvar = self(img)
                ### return z
                loss = self.lossfunction(img, pred, beta, mu, logvar)
                # Step the optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # housekeeping
                losses.append(loss.item())

                if i % 100 == 0:
                    print("{}/{}, loss: {}".format(i, len(dataset), np.mean(losses[-100:])))
                    show_image(img.cpu().numpy(), pred.detach().view(-1, 28, 28).squeeze(0).cpu().numpy())
            print("Epoch {}/{} done, loss: {}!".format(e+1, epochs, np.mean(losses)))

vae = VAE()

# Create dataset (downloads MNIST dataset to data folder)
#dataset_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
dataset_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

# Create dataloader object. Can load batches of data from the dataset and apply shuffling or other functions to them.
data_loader = torch.utils.data.DataLoader(dataset_train,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=1)

# Hyperparameter
# - beta: How similar the latent distribution given my mu and logvar has to be to a gaussian unit distribution
# - learning rate (see optimizer)
# - batch size: how many images per forward 
# - Network architecture
vae.train(data_loader, epochs=10, beta=0.000)
