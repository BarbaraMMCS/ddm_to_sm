from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import tensor, float32

from preprocessing.config import CYNGNSS_FEATURES_HEADER
from training.autoencoder import AUTOENCODER, CONV_AUTOENCODER

import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from typing import Union
import time


RANDOM_SEED = 49
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
NUM_EPOCHS = 30
NUM_CLASSES = 2

optimizer = torch.optim.Adam(AUTOENCODER.parameters(), lr=LEARNING_RATE)

class DDMDatasets(Dataset):
    def __init__(self, dataset_path: Union[str, Path]):
        print(f"Loading dataset from {dataset_path}")
        start = time.time()
        data = np.load(dataset_path)

        ddms = data[:,data.shape[1]-(17*11):]
        maxes = np.amax(ddms, axis=1).reshape(len(ddms), 1)
        mines = np.amin(ddms, axis=1).reshape(len(ddms), 1)
        ddms = (ddms - mines) / (maxes - mines)
        self.ddms = ddms.astype('float32')
        end = time.time()
        print(f"Done loading dataset from {dataset_path} ({end - start:.2f}s)")

    def __len__(self):
        return self.ddms.shape[0]

    def __getitem__(self, idx):
        # return self.ddms[idx, :, :] # tensor(self.ddms[idx, :, :], dtype=float32),
        # return tensor(self.ddms[idx, :, :]).to("cuda")
        return torch.from_numpy(self.ddms[idx, :]).cuda()

class ConvDDMDatasets(Dataset):
    def __init__(self, dataset_path: Union[str, Path]):
        print(f"Loading dataset from {dataset_path}")
        start = time.time()
        data = np.load(dataset_path)

        ddms = data[:, data.shape[1] - (17 * 11):]
        maxes = np.amax(ddms, axis=1).reshape(len(ddms), 1)
        mines = np.amin(ddms, axis=1).reshape(len(ddms), 1)
        ddms = (ddms - mines) / (maxes - mines)
        self.ddms = ddms.reshape(ddms.shape[0], 1, 17, 11).astype('float32')
        end = time.time()
        print(f"Done loading dataset from {dataset_path} ({end - start:.2f}s)")

    def __len__(self):
        return self.ddms.shape[0]

    def __getitem__(self, idx):
        # return self.ddms[idx, :, :] # tensor(self.ddms[idx, :, :], dtype=float32),
        # return tensor(self.ddms[idx, :, :]).to("cuda")
        return torch.from_numpy(self.ddms[idx, :, :, :]).cuda()

def train_autoencoder(num_epochs, model, optimizer,
                         train_loader, loss_fn=None,
                         logging_interval=100,
                         save_model=None):
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}
    if loss_fn is None:
        loss_fn = F.mse_loss
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, features in enumerate(train_loader):
            # FORWARD AND BACK PROP
            logits = model(features)
            loss = loss_fn(logits, features)
            optimizer.zero_grad()
            loss.backward()
            # UPDATE MODEL PARAMETERS
            optimizer.step()
            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch + 1, num_epochs, batch_idx,
                         len(train_loader), loss))
        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model + '.pth')
        torch.save(model.encoder.state_dict(), save_model + 'encoder.pth')
    return log_dict




if __name__ == '__main__':
    # AUTOENCODER.cuda()
    # d = DDMDatasets('../data/train_label/20220101.npy')
    # train_loader = DataLoader(dataset=d,
    #                           batch_size=250,
    #                           num_workers=0,
    #                           shuffle=True)
    # log_dict = train_autoencoder(num_epochs=100, model=AUTOENCODER,
    #                              optimizer=optimizer,
    #                              train_loader=train_loader,
    #                              logging_interval=250,
    #                              save_model='normal_')

    CONV_AUTOENCODER.cuda()
    conv_d = ConvDDMDatasets('../data/train_label/20220101.npy')
    conv_train_loader = DataLoader(dataset=conv_d,
                              batch_size=250,
                              num_workers=0,
                              shuffle=True)
    log_dict = train_autoencoder(num_epochs=100, model=CONV_AUTOENCODER,
                                 optimizer=optimizer,
                                 train_loader=conv_train_loader,
                                 logging_interval=250,
                                 save_model='conv_')

