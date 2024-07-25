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
metric_funcs= [snr, rmse, mae, rrse, nrmse, trac]

# put metric data for each model in a table
# models are: pure physics, pure nn, indirect, pinn, pinn control, 
# delta model 1, delta learning, delta learning control, informed architecture
metrics_table = np.zeros((9, len(metric_funcs)))
#%% pure physics
m = 1
c = 0.2
all_data = np.load('./data/v4/with_friction.npy')[:,:-1,:]

t = all_data[0,:,0]
x = all_data[:,:,1]
v = all_data[:,:,2]
a = all_data[:,:,3]
k = all_data[:,:,4]
F = all_data[:,:,5]
k_pred = np.load('./model_predictions/pure_physics/k_pred.npy')[:,:-1]

metrics_table[0] = np.array([m(k, k_pred) for m in metric_funcs])

# cumulatitive RMSE error over time for all experiments
rmse_t = np.sqrt(np.sum((k - k_pred)**2, axis=0))
residual = np.sum(np.abs(F - m*a - c*v - k*x), axis=0)
np.save('./metric_results/pure_physics/rmse_t.npy', rmse_t)
np.save('./metric_results/pure_physics/residual.npy', residual)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, rmse_t)
ax1.set_ylabel('RMSE (N/s)')
ax2.plot(t, residual)
ax2.set_xlabel('time (s)')
ax2.set_ylabel('residual (N)')
plt.tight_layout()
#%% pure nn
all_data = np.load('./data/v4/with_friction.npy')
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]

t = all_data[0,:,0]
a = all_data[:,:,3]
k = all_data[:,:,4]

k_test = k[80:,49:]

k_pred = np.load('./model_predictions/pure_nn/k_pred.npy')

metrics_table[1] = np.array([m(k_test, k_pred) for m in metric_funcs])
#%% indirect
all_data = np.load('./data/v4/with_friction.npy')
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]

t = all_data[0,:,0]
a = all_data[:,:,3]
k = all_data[:,:,4]

k_test = k[80:,49:]

k_pred = np.load('./model_predictions/indirect/k_pred.npy')

metrics_table[2] = np.array([m(k_test, k_pred) for m in metric_funcs])
#%% pinn (and control)
test_data = np.load('./data/v4/pinn_test.npy').T
# downsample by a factor of 20 so that sampling rate it 50 S/s
test_data = test_data[:,::20]
t = test_data[0,1:]
x = test_data[1,1:]
k = test_data[5,1:]
pinn_pred = np.load('./model_predictions/pinn/pred_out.npy')
x_pinn = pinn_pred[:,0]

plt.figure()
plt.plot(t, x, label='true')
plt.plot(t, x_pinn, label='pinn x')

k_pinn = pinn_pred[:,1]
k_control = np.load('./model_predictions/pinn/control_out.npy')

metrics_table[3] = np.array([m(k, k_pinn) for m in metric_funcs])
metrics_table[4] = np.array([m(k, k_control) for m in metric_funcs])
#%% delta learning (and control)
with_friction_data = np.load('./data/v4/with_friction.npy')
# downsample by a factor of 20 so that sampling rate it 50 S/s
with_friction_data = with_friction_data[:,:-1:20,:]

t_wf = with_friction_data[0,:,0]
x_wf = with_friction_data[:,:,1]
v_wf = with_friction_data[:,:,2]
a_wf = with_friction_data[:,:,3]
k_wf = with_friction_data[:,:,4]
F_wf = with_friction_data[:,:,5]

model_1_pred = np.load('./model_predictions/delta_learning/with_friction_model_1.npy')
delta_model_pred = np.load('./model_predictions/delta_learning/with_friction_combined_model.npy')
delta_control_pred = np.load('./model_predictions/delta_learning/with_friction_control.npy')

metrics_table[5] = np.array([m(k_wf, model_1_pred) for m in metric_funcs])
metrics_table[6] = np.array([m(k_wf, delta_model_pred) for m in metric_funcs])
metrics_table[7] = np.array([m(k_wf, delta_control_pred) for m in metric_funcs])
#%% informed structure
all_data = np.load('./data/v4/with_friction.npy')
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]
k = all_data[:,:,4]
k_test = k[80:]

k_pred = np.load('./model_predictions/informed_structure/k_pred.npy').squeeze()
metrics_table[8] = np.array([m(k_test, k_pred) for m in metric_funcs])
#%%
np.save('./metric_results/metrics_table')

