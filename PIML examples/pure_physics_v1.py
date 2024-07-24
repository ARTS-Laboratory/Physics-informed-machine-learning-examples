import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
"""
Using only a physics understanding of the system when calculating k
"""
#%%
'''
Function to solve the spring-mass system. Solved with RK4.
'''
def spring_mass_system(t, m, c, k, F, x0, xdot0, dt):
    def f(xdot, x, Fi):
        xddot = -c*xdot - k*x + Fi
        return xddot, xdot
    y = np.zeros(t.size)
    xdot = xdot0
    x = x0
    for i in range(len(y)):
        xdot1, x1 = f(xdot, x, F[i])
        xdot2, x2 = f(xdot + 0.5*dt*xdot1, x + 0.5*dt*x1, F[i])
        xdot3, x3 = f(xdot + 0.5*dt*xdot2, x + 0.5*dt*x2, F[i])
        xdot4, x4 = f(xdot + dt*xdot3, x + dt*x3, F[i])
        
        xdot = xdot + dt/6*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4)
        x = x + dt/6*(x1 + 2*x2 + 2*x3 + x4)
        xddot = -c*xdot - k*x + F[i]
        # print(xddot.shape)
        y[i] = xddot
    return y
    
def spring_mass_system1(t, m, c, k, F, x0, xdot0, dt):
    def f(xdot, x, Fi):
        xddot = (-c*xdot - k*x + Fi)/m
        # print(xddot)
        return xddot, xdot
    y = np.zeros((t.size, 3))
    xdot = xdot0
    x = x0
    for i in range(len(y)):
        xdot1, x1 = f(xdot, x, F[i])
        xdot2, x2 = f(xdot + 0.5*dt*xdot1, x + 0.5*dt*x1, F[i])
        xdot3, x3 = f(xdot + 0.5*dt*xdot2, x + 0.5*dt*x2, F[i])
        xdot4, x4 = f(xdot + dt*xdot3, x + dt*x3, F[i])
        
        xdot = xdot + dt/6*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4)
        x = x + dt/6*(x1 + 2*x2 + 2*x3 + x4)
        xddot = (-c*xdot - k*x + F[i])/m
        # print(xddot)
        y[i] = [xddot, xdot, x]
    return y    

'''
Extract the expected k from the system
m: scalar value of mass
c: scalar value of damping
a: array of acceleration values
dt: time between acceleration samples
'''
# def extract_k(m, c, a, F, dt):
#     N = a.size
#     t = np.arange(0, st)
#     # freq = fftfreq(N, dt)
#     # y = np.abs(fft(a))
#     # # collect positive frequencies
#     # freq = freq[:N//2]
#     # y = y[:N//2]
    
#     f = lambda x: spring_mass_system()
    
#     plt.figure()
#     plt.plot(freq, y)
#%% load dataset
all_data = np.load('./data/v2/all_data.npy')

t = all_data[0,:,0]
x = all_data[:,:,1]
v = all_data[:,:,2]
a = all_data[:,:,3]
k = all_data[:,:,4]
F = all_data[:,:,5]

dt = t[1] - t[0]
N = int(1/dt) # take 1 second for a sample
m = 1
c = 0.2
forced_freq = 5

# k_bounds = [1500, 500]
#%% test RK4 algorithm
# x0 = 0
# v0 = 1
# k=1500

# t_batch = t[0:1000]
# F_batch = np.zeros(t_batch.shape)


# reconstructed_data = spring_mass_system1(t_batch, m, c, k, F_batch, x0, v0, dt)
# a_r = reconstructed_data[:,0]
# v_r = reconstructed_data[:,1]
# x_r = reconstructed_data[:,2]

# plt.figure()
# # plt.plot(t_batch, a_r, label='acceleration')
# plt.plot(t_batch, v_r, label='velocity')
# plt.plot(t_batch, x_r, label='position')
# plt.legend()
# plt.xlim((t_batch[0], t_batch[-1]))
# plt.tight_layout()
k_pred_tot = np.zeros(k.shape)
#%%
for test in range(x.shape[0]):
    
    a_test = a[test]
    F_test = F[test]
    a_batches = a_test[1:].reshape(-1, N)
    t_batches = t[1:].reshape(-1, N)
    F_batches = F_test[1:].reshape(-1, N)
    x0_batches = x[test,::N]
    v0_batches = v[test,::N]
    n_batches = a_batches.shape[0]
    k_pred = np.zeros(a_test.size)
    
    x0_batch = x0_batches[0]
    v0_batch = v0_batches[0]
    for j in range(n_batches):
        #%%
        a_batch = a_batches[j]
        t_batch = t_batches[j]
        F_batch = F_batches[j]
        # x0_batch = x0_batches[j]
        # v0_batch = v0_batches[j]
        
        f = lambda t, k: spring_mass_system(t, m=m, c=c, k=k, F=F_batch, x0=x0_batch, xdot0=v0_batch, dt=dt)
        f1 = lambda t, k: spring_mass_system1(t, m=m, c=c, k=k, F=F_batch, x0=x0_batch, xdot0=v0_batch, dt=dt)
        
        try:
            k_param, _ = curve_fit(f, t_batch, a_batch, bounds=[500, 1500])
            k_param = k_param[0]
        except: # If optimal parameter is not found and error is thrown
            k_param = 0
        
        k_pred[N*j:N*(j+1)] = k_param
        
        a_reconstructed = f(t_batch, k_param)
        
        reconstructed_data = f1(t_batch, k_param)
        a_r = reconstructed_data[:,0]
        v_r = reconstructed_data[:,1]
        x_r = reconstructed_data[:,2]
        
        # plt.figure()
        # plt.plot(t_batch, a_batch, label='true')
        # plt.plot(t_batch, a_reconstructed, label='reconstruced')
        # # plt.plot(t_batch, a_r, label='reconstruced')
        # plt.plot(t_batch, F_batch)
        # plt.legend()
        # plt.xlim((t_batch[0], t_batch[-1]))
        # plt.tight_layout()
        
        # plt.figure()
        # plt.plot(t_batch, v_r, label='velocity')
        # plt.plot(t_batch, x_r, label='position')
        # plt.legend()
        # plt.xlim((t_batch[0], t_batch[-1]))
        # plt.tight_layout()
        
        x0_batch = x_r[-1]
        v0_batch = v_r[-1]
    
    plt.figure()
    plt.plot(t, k_pred, label='pred')
    plt.plot(t, k[test], label='true')
    plt.legend()
    plt.tight_layout()
    
    k_pred_tot[test] = k_pred
    print('finished test #%d'%(test+1))
np.save('./model_predictions/pure_physics/k_pred.npy', k_pred_tot)