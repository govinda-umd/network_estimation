#!/usr/bin/env python
# coding: utf-8

# # Sep 10, 2023: pytorch lightning

# In[1]:


import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


# In[2]:


encoder = nn.Sequential(
    nn.Linear(28*28, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)
decoder = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 28*28)
)

class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log('val_loss', val_loss, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = nn.functional.mse_loss(x_hat, x)
        self.log('test_loss', test_loss, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
autoencoder = AutoEncoder(encoder, decoder)


# In[3]:


train_set = MNIST(os.getcwd(), download=False, train=True, transform=ToTensor())
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size
seed = torch.Generator().manual_seed(42)
train_set, valid_set = utils.data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

test_set = MNIST(os.getcwd(), download=False, train=False, transform=ToTensor())

train_loader = utils.data.DataLoader(train_set, num_workers=10)
valid_loader = utils.data.DataLoader(valid_set, num_workers=10)
test_loader = utils.data.DataLoader(test_set, num_workers=10)


# In[4]:


trainer = pl.Trainer(
    limit_train_batches=100, 
    max_epochs=1, 
    accelerator='cpu',
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
)
train_result = trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)


# In[5]:


test_result = trainer.test(model=autoencoder, dataloaders=test_loader)
test_result[0]['test_loss']


# In[6]:


encoder = autoencoder.encoder
encoder.eval()

fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

