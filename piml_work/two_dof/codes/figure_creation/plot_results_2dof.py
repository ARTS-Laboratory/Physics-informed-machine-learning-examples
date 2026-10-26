#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt



"""
Plots

Code written by Nile Coble

"""

plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
                                    'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
                                    'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
                                    'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.it': 'serif:italic'})
plt.rcParams.update({'mathtext.bf': 'serif:bold'})
plt.close('all')

#%% load data
all_data=np.load("./data/two_dof_decay_wf.npz")['data']
t=all_data[0,:,0]
F=all_data[:,:,11]

# first mass
x1=all_data[:,:,1]
v1=all_data[:,:,2]
a1=all_data[:,:,3]

# second mass
x2=all_data[:,:,5]
v2=all_data[:,:,6]
a2=all_data[:,:,7]

# stiffness
k1=all_data[:,:,8]
k2=all_data[:,:,9]
k3=all_data[:,:,10]

#%% acceleration plot
i=0
plt.figure(figsize=(6.5,3.5))
plt.plot(t,a1[i],linewidth=0.7)
plt.plot(t,a2[i],linewidth=0.7)
plt.xlim([0,180])
plt.ylim([-3,3])
plt.xlabel("time (s)")
plt.ylabel('acceleration (m/s$^2$)')

#%% velocity plot
i=0
plt.figure(figsize=(6.5,3.5))
plt.plot(t,v1[i],linewidth=0.7)
plt.plot(t,v2[i],linewidth=0.7)
plt.xlim([0,180])
plt.ylim([-.125,.125])
plt.xlabel("time (s)")
plt.ylabel('velocity (m/s)')

#%% acceleration plot
i=0
plt.figure(figsize=(6.5,3.5))
plt.plot(t,x1[i],linewidth=0.7)
plt.plot(t,x2[i],linewidth=0.7)
plt.xlim([0,180])
plt.ylim([-.004,.004])
plt.xlabel("time (s)")
plt.ylabel('displacement (m)')

#%% stiffness plot
plt.figure(figsize=(6.5,2.5))
plt.plot(t,k1.T)
plt.xlim([0,180])
plt.ylim([450,1550])
plt.xlabel("time (s)")
plt.ylabel('stiffness (N/m)')

plt.figure(figsize=(6.5,2.5))
plt.plot(t,k2.T)
plt.xlim([0,180])
plt.ylim([450,1550])
plt.xlabel("time (s)")
plt.ylabel('stiffness (N/m)')

plt.figure(figsize=(6.5,2.5))
plt.plot(t,k3.T)
plt.xlim([0,180])
plt.ylim([450,1550])
plt.xlabel("time (s)")
plt.ylabel('stiffness (N/m)')