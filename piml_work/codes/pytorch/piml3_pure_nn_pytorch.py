#%% Installs the code
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import tqdm

import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#%% main code
def main():
    # loads data
    all_data = np.load("./data/sigmoid_wf_90s_3m.npz")['data']
    # downsamples data (6000 data_points per sample)
    all_data = all_data[:,:-1:20,:]
    
    t = all_data[0,:,0] # time (s)
    x = all_data[:,:,1] # displacement (m)
    v = all_data[:,:,2] # velocity (m/s)
    a = all_data[:,:,3] # acceleration (m/s^2)
    k = all_data[:,:,4] # stiffness (N/m)
    F = all_data[:,:,5] # force (N)
    
    N=t.shape[0]
    
    # splits data into testing and training batches (80-20 split)
    a_train,a_test,k_train,k_test=train_test_split(a,k,train_size=.8,shuffle=True)
    

    
    a_scaler = StandardScaler()
    k_scaler = StandardScaler()

    a_scaler.fit(a_train)
    k_scaler.fit(k_train)
    
    a_train=torch.tensor(a_scaler.transform(a_train),dtype=torch.float32)
    a_test=torch.tensor(a_scaler.transform(a_test),dtype=torch.float32)
    
    k_train=torch.tensor(k_scaler.transform(k_train),dtype=torch.float32)
    k_test=torch.tensor(k_scaler.transform(k_test),dtype=torch.float32)
    
    
    # model (3 hidden layers with 100 neurons each, sigmoids)
    model=nn.Sequential(
        nn.Linear(N,100),
        nn.Sigmoid(),
        nn.Linear(100,100),
        nn.Sigmoid(),
        nn.Linear(100,100),
        nn.Sigmoid(),
        nn.Linear(100, N))
    loss_fn=nn.MSELoss() # loss function 
    optimizer=optim.Adam(model.parameters(),lr=.001,betas=(.9,.999)) # optimization function
    
    # training parameters
    n_epochs = 50 # number of epochs
    batch_size= 32 # number of batches
    
    batch_start = torch.arange(0,len(a_train), batch_size) #batch
    best_mse = np.inf # default Best MSE
    k_pred_tot=[]
    history=[] # creates a list plotting RMSE
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", 
                       mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # Nueral Network Layer
                a_batch = a_train[start:start + batch_size]
                k_batch = k_train[start:start + batch_size]

                k_pred = model(a_batch)
                
                # Loss
                loss = loss_fn(k_pred, k_batch)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # evaluates model
        model.eval()
        with torch.no_grad():
            k_pred = model(a_test) # predicts stiffness from training data
            mse = float(loss_fn(k_pred,k_test))
        k_pred=k_scaler.inverse_transform(k_pred)
        k_pred_tot.append(k_pred)
        if mse < best_mse: # checks is best_mse should be updated
            best_mse = mse
            best_state=copy.deepcopy(model.state_dict())
        history.append(np.sqrt(mse))
    print(f"Final MSE: {best_mse:.6f}")
    print(f'Final RSME: {np.sqrt(best_mse):.6f}')
    
    k_test=k_scaler.inverse_transform(k_test)
    k_pred_tot.append(k_pred)
    model.load_state_dict(best_state)
    torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_validation_mse_normalized": best_mse,
    "input_length": N,}, "model_saves/pure_nn/stiffness_model.pt")
    
    np.savez(
    "model_predictions/pure_nn/test_results.npz",
    time=t,
    k_true=k_test,
    k_pred=k_pred_tot,
    validation_rmse_history=history,
    )
    
    # plot data
    i=4
    T=round(max(t)) # rounds maximum time
    
    plt.figure(figsize=(6.5,3.5))
    plt.plot(t,k_pred_tot[19][i],label='neural network model')
    plt.plot(t,k[i],label='actual data')
    plt.xlim([0,T])
    plt.xlabel('time (s)')
    plt.ylabel('stiffness (N/m)')
    plt.tight_layout()
    plt.savefig('./figures/sigmoid.png')
    
    
    # RSME over time
    x_epoch=np.linspace(0,n_epochs,n_epochs)
    plt.figure()
    plt.plot(x_epoch,history)
    plt.title('RMSE over each epoch')
    plt.xlim([0,n_epochs])
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.tight_layout()

if __name__=='__main__':
    main()