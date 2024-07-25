import numpy as np
import matplotlib.pyplot as plt
"""
Functions for metric results.
"""

""" signal to noise ratio """
def snr(sig, pred, dB=True):
    noise = sig - pred
    a_sig = np.sqrt(np.mean(np.square(sig)))
    a_noise = np.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*np.log10(snr)
""" root mean squared error """
def rmse(sig, pred, squared=False):
    error = sig - pred
    num = np.sum(np.square(error))
    denom = np.size(sig)
    e = num/denom
    if(squared):
        return e
    return np.sqrt(e)
""" root relative squared error """
def rrse(sig, pred):
    error = sig - pred
    mean = np.mean(sig)
    num = np.sum(np.square(error))
    denom = np.sum(np.square(sig-mean))
    return np.sqrt(num/denom)
""" normalized root mean squared error """
def nrmse(sig, pred):
    return rmse(sig, pred)/(np.max(sig)-np.min(sig))
""" time response assurance criterion """
def trac(sig, pred):
    num = np.square(np.sum(sig * pred))
    denom = np.sum(sig * sig) * np.sum(pred * pred)
    return num/denom
""" mean absolute error """
def mae(sig, pred):
    return np.sum(np.abs(sig-pred))/sig.size
#%%
metric_funcs= [snr, rmse, rrse, nrmse, trac, mae]
#%% pure physics
m = 1
c = 0.2
all_data = np.load('./data/v4/with_friction.npy')

t = all_data[0,:,0]
x = all_data[:,:,1]
v = all_data[:,:,2]
a = all_data[:,:,3]
k = all_data[:,:,4]
F = all_data[:,:,5]
k_pred = np.load('./model_predictions/pure_physics/k_pred.npy')

metrics = np.array([m(k, k_pred) for m in metric_funcs])

# cumulatitive RMSE error over time for all experiments
rmse_t = np.sqrt(np.sum((k - k_pred)**2, axis=0))
residual = np.sum(np.abs(F - m*a - c*v - k*x), axis=0)
np.save('./metric_results/pure_physics/rmse_t.npy', rmse_t)
np.save('./metric_results/pure_physics/residual.npy', residual)
np.save('./metric_results/pure_physics/metrics.npy', metrics)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, rmse_t)
ax1.set_ylabel('RMSE (N/s)')
ax2.plot(t, residual)
ax2.set_ylabel('residual (N)')
plt.tight_layout()
#%% pure nn
all_data = np.load('./data/v4/with_friction.npy')
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]

t = all_data[0,:,0]
a = all_data[:,:,3]
k = all_data[:,:,4]

k_test = k[80:,50:]

k_pred = np.load('./model_predictions/pure_nn/k_pred.npy')

metrics = np.array([m(k, k_pred) for m in metric_funcs])
#%% indirect
all_data = np.load('./data/v4/with_friction.npy')
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]

t = all_data[0,:,0]
a = all_data[:,:,3]
k = all_data[:,:,4]

k_test = k[80:,50:]

k_pred = np.load('./model_predictions/indirect/k_pred.npy')

metrics = np.array([m(k, k_pred) for m in metric_funcs])