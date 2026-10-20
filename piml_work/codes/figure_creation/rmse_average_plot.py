#%% imports all modules
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'text.usetex': 'True'})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
                                       'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
                                       'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman',
                                       'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.rm': 'serif'})

#%%  load all data
all_data = np.load('./data/mathieu_k_mean_0_4hz_wf.npz')['data']
    
    
t = all_data[0,:,0]
x = all_data[:,:,1]
v = all_data[:,:,2]
a = all_data[:,:,3]
k = all_data[:,:,4]
F = all_data[:,:,5]
    
t_max=np.round(max(t)) # rounds to the nearest second
    
data_pred_tot=np.load('./model_predictions/pure_physics/k_pred_mathieu_k_mean_0_4hz_wf.npy')
    
k_true=k
k_pred=data_pred_tot[:,0,:]
    
k_rmse_all=np.sqrt((k_true-k_pred)**2)
k_rmse_av=np.sqrt(np.mean((k_true-k_pred)**2,axis=0))
k_rmse_std=np.sqrt(np.std((k_true-k_pred)**2,axis=0))
    
k_rmse_std_a=k_rmse_av+k_rmse_std
k_rmse_std_b=k_rmse_av-k_rmse_std

#%% pure physics stiffness comparison
i=12
plt.figure(figsize=(6.5,2.5))
plt.plot(t,k_true[i])
plt.plot(t,k_pred[i])
plt.xlabel('time (s)')
plt.ylabel('RMSE')
plt.xlim([0,t_max])
plt.ylim([450,1550])
plt.tight_layout()
plt.savefig('./figures/stiffness_comparison_k_pred_mathieu_k_mean_0_4hz_wf.png',dpi=300)


#%% plot of all rmse plots
plt.figure(figsize=(6.5,2.5))
plt.plot(t,k_rmse_all.T)
# plt.title('RMSE plot of all trajectories')
plt.xlabel('time (s)')
plt.ylabel('RMSE')
plt.xlim([0,t_max])
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('./figures/all_pure_phys_mathieu_k_mean_0_4hz_wf.png',dpi=300)
    
#%% plot of all rmse plots with average error
plt.figure(figsize=(6.5,2.5))
plt.plot(t,k_rmse_all.T,alpha=.25)
plt.plot(t,k_rmse_av,linewidth=2.5,color='black')
# plt.title('RMSE of all trajectories with average')
plt.xlabel('Time (s)')
plt.ylabel('RMSE')
plt.xlim([0,t_max])
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('./figures/pure_phys_k_rmses_mathieu_k_mean_0_4hz_wf.png',dpi=300)
    
#%% plot of standard deviation of rmse plots
plt.figure(figsize=(6.5,2.5))
plt.plot(t,k_rmse_std)
# plt.title('standard deviation of RMSE of all trajectories')
plt.xlabel('Time (s)')
plt.ylabel('RMSE')
plt.xlim([0,t_max])
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('./figures/pure_phys_k_rmse_std_mathieu_k_mean_0_4hz_wf.png',dpi=300)

#%% mean +- std
fig, ax=plt.subplots(1,1,sharex=True,figsize=(6.5,2.5))
ax.fill_between(t,k_rmse_std_a,k_rmse_std_b,alpha=.3)
ax.plot(t,k_rmse_av)
# plt.title('mean and standard deviation of RMSE of trajectories')
plt.xlabel('Time (s)')
plt.ylabel('RMSE')
plt.xlim([0,t_max])
plt.ylim([0,1200])
plt.tight_layout()
plt.savefig('./figures/pure_phys_k_rmse_one_std_mathieu_k_mean_0_4hz_wf.png',dpi=300)
