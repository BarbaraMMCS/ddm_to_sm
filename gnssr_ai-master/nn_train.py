import copy
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split

def train_neural_network(data_frame):
    # Prepare data
    X = data_frame.drop(columns=['smap_sm_d'])
    y = data_frame['smap_sm_d']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train.to_numpy().astype(np.float32), dtype=torch.float32).cuda()
    y_train = torch.tensor(y_train.to_numpy().astype(np.float32), dtype=torch.float32).cuda().reshape(-1, 1)
    X_test = torch.tensor(X_test.to_numpy().astype(np.float32), dtype=torch.float32).cuda()
    y_test = torch.tensor(y_test.to_numpy().astype(np.float32), dtype=torch.float32).cuda().reshape(-1, 1)

    # Define the model
    model = nn.Sequential(
        nn.Linear(34, 16), nn.ReLU(),
        nn.Linear(16,8), nn.ReLU(),
        nn.Linear(8, 1)
    ).to(device=0)

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training parameters
    n_epochs = 500
    patience = 30
    batch_size = 120_000
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Training loop
    best_ubrmse = np.inf
    best_weights = None
    history = []
    training_ubrmse = []
    no_improvement_count = 0
    start_t = time()
    print('Training...')
    try:
        for epoch in range(n_epochs):
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    X_batch = X_train[start:start + batch_size]
                    y_batch = y_train[start:start + batch_size]
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    bar.set_postfix(mse=float(loss))

            # Evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_test)
            error = (y_test - y_pred)
            mse = float(loss_fn(y_pred, y_test))
            rmse = np.sqrt(mse)
            bias = float(error.mean())
            ubrmse = np.sqrt(rmse ** 2 - bias ** 2)
            history.append(ubrmse)

            y_pred_train = model(X_train)
            error_train = (y_train - y_pred_train)
            mse_train = float(loss_fn(y_pred_train, y_train))
            rmse_train = np.sqrt(mse_train)
            bias_train = float(error_train.mean())
            ubrmse_train = np.sqrt(rmse_train ** 2 - bias_train ** 2)
            training_ubrmse.append(ubrmse_train)

            if ubrmse < best_ubrmse:
                best_ubrmse = ubrmse
                best_weights = copy.deepcopy(model.state_dict())
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f'Early stopping at epoch {epoch} as validation performance did not improve for {patience} epochs.')
                break

            if epoch % 10 == 0:
                torch.save(model, f'models/1M_nn_2020_{epoch}.pt')
    finally:
        print('Training: %.3f seconds' % (time() - start_t))

        # Restore model and return best accuracy
        model.load_state_dict(best_weights)
        torch.save(model, 'models/1M_best_nn_2020.pt')
        print("ubrMSE: %.2f" % best_ubrmse)
        idx = 0
        plt.plot(history[idx:], label='Validation ubRMSE')
        plt.plot(training_ubrmse[idx:], label='Training ubRMSE')
        plt.ylabel('ubRMSE')
        plt.xlabel(f'epochs')
        plt.ylim(0, 0.2)
        plt.legend()
        plt.savefig('figures/1M_nn_ubrmse_val.svg')


if __name__ == '__main__':
    path = 'dataset/nn/sm_known_2020_9.0km.csv'
    print(f'Read: {path}')
    start = time()
    df = pd.read_csv(path)
    print('%.3f seconds' % (time() - start))

    train_neural_network(df)