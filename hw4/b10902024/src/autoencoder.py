import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.utils.data import DataLoader, TensorDataset

def plot_learning_curve(curve, path):
    plt.figure()
    plt.plot(curve, label="train_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join("./result", path))
    plt.close()

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
    
    def forward(self, x):
        #TODO: 5%
        return self.decoder(self.encoder(x))
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 5%
        # dataloader
        dataloader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float)), batch_size=batch_size, shuffle=False, drop_last=False)
        
        # loss & optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # for ploting
        train_loss_list = []

        for epoch in range(epochs):
            train_loss = 0.0
            for batch, data in enumerate(dataloader):
                batch_data = torch.cat(data)
                # forward pass
                pred = self.forward(batch_data)
                # calculate loss
                loss = criterion(pred, batch_data)
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # evaluate
                train_loss += loss.item()
            train_loss /= len(dataloader.dataset)
            print()
            print(f'[{epoch + 1}/{epochs}] Train Loss: {train_loss:.5f}')
            train_loss_list.append(train_loss)
        # plot_learning_curve(train_loss_list, "ae_Adam_thin.png")
    
    def transform(self, X):
        #TODO: 2%
        return self.encoder(torch.tensor(X, dtype=torch.float)).detach().numpy()
    
    def reconstruct(self, X):
        #TODO: 2%
        return self.decoder(torch.tensor(self.transform(X), dtype=torch.float)).detach().numpy()


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        #TODO: 3%
        return x + torch.normal(mean=0, std=self.noise_factor, size=x.shape)
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 4%
        # dataloader
        dataloader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float)), batch_size=batch_size, shuffle=False, drop_last=False)
        
        # loss & optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # for ploting
        train_loss_list = []

        for epoch in range(epochs):
            train_loss = 0.0
            for batch, data in enumerate(dataloader):
                batch_data = torch.cat([self.add_noise(x) for x in data])
                # forward pass
                pred = self.forward(batch_data)
                # calculate loss
                loss = criterion(pred, batch_data)
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # evaluate
                train_loss += loss.item()
            train_loss /= len(dataloader.dataset)
            print()
            print(f'[{epoch + 1}/{epochs}] Train Loss: {train_loss:.5f}')
            train_loss_list.append(train_loss)
        # plot_learning_curve(train_loss_list, "deno_ae_Adam_thin.png")
