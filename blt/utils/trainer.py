import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from utils.util import models_save, get_deltas, AvgMeter

def train_bootstrapped(model_type, dataset, model, logdir, obs_idxs, skew, \
                     num_epochs=500, batch_size=32, checkpoint_path=None, store_train_deltas=True, similarity_type=None, pct=.25):
    dataset_bs = dataset.copy()
    idx = np.random.randint(len(dataset['train_Y']), size = int(len(dataset['train_Y'])*pct))
    dataset_bs['train_X'] = dataset_bs['train_X'][idx]
    dataset_bs['train_Y'] = dataset_bs['train_Y'][idx]
    return train_supervised(model_type, dataset_bs, model, logdir, obs_idxs, skew,
                     num_epochs, batch_size, checkpoint_path, store_train_deltas, similarity_type), idx

def train_supervised(model_type, dataset, model, logdir, obs_idxs, skew, \
                     num_epochs=500, batch_size=32, checkpoint_path=None, store_train_deltas=True, similarity_type=None):
    """train model"""
    
    X, Y = dataset['train_X'], dataset['train_Y']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(list(model.parameters()))
    epoch_losses = []
    idxs = np.array(range(len(X)))
    num_batches = len(idxs) // batch_size
    if len(idxs) < batch_size:
        num_batches = 1
        batch_size = len(idxs)
    print('num_epochs', num_epochs, 'num_batches', num_batches)
    train_deltas = []
    
    # Train the model with regular SGD
    for epoch in range(num_epochs):
        loss_meter = AvgMeter()
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            batch_gradients, batch_params = [], []
            optimizer.zero_grad()

            t1_idx = np.random.randint(len(X), size=(batch_size,)) # Indices of A
            t2_idx = np.random.randint(len(X), size=(batch_size,)) # Indices of sample B
            # priviledged training, knowledge of the Y distribution's skewness (transformation type/object class type)
            if skew=='right': #t2 have higher ys than t1
                swap_idxs = (Y[t1_idx] > Y[t2_idx]).flatten()
                t1_idx[swap_idxs], t2_idx[swap_idxs] = t2_idx[swap_idxs], t1_idx[swap_idxs]
            elif skew=='left':
                swap_idxs = (Y[t1_idx] < Y[t2_idx]).flatten()
                t1_idx[swap_idxs], t2_idx[swap_idxs] = t2_idx[swap_idxs], t1_idx[swap_idxs]
                
            t1_X = torch.Tensor(np.concatenate([X[c_idx][obs_idxs][None] for c_idx in t1_idx])).float().to(device)
            t1_Y = torch.Tensor(np.concatenate([Y[c_idx][None] for c_idx in t1_idx])).float().to(device)
            t2_X = torch.Tensor(np.concatenate([X[c_idx][obs_idxs][None] for c_idx in t2_idx])).float().to(device)
            t2_Y = torch.Tensor(np.concatenate([Y[c_idx][None] for c_idx in t2_idx])).float().to(device)

            if model_type == 'mlp':
                y1_pred = model(t1_X)
                loss = torch.mean(torch.linalg.norm(y1_pred - t1_Y, dim=-1))
            elif 'bilinear' in model_type:
                deltas = get_deltas(t1_X, t2_X, similarity_type)
                if store_train_deltas:
                    delta_idx = np.random.randint(len(deltas), size=(1,))[0] # store 1 delta every batch
                    train_deltas.append(deltas[delta_idx].cpu().detach().numpy())
                y2_pred = model(t1_X, deltas)
                loss = torch.mean(torch.linalg.norm(y2_pred - t2_Y, dim=-1)) # MAE
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_meter.update(loss.item(), batch_size)            
            if (i+1) % 5 == 0:
                print('[%d, %5d] loss: %.8f' %
                    (epoch+1, i+1, running_loss/(i+1)))

        epoch_losses.append(running_loss/num_batches)
        if (epoch+1) % 2000 == 0 and checkpoint_path:
            models_save(model, logpath=osp.join(checkpoint_path, str(epoch)+'.pt'))
        

    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('MAE')
    plt.plot(epoch_losses)
    plt.yscale('log')
    plt.savefig(os.path.join(logdir, model_type+'_losses.png'))
    print('Finished Training')

    return model, np.array(train_deltas).reshape(-1,len(obs_idxs))