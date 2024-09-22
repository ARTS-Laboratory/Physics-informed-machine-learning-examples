import numpy as np
import os
#%% load data
# n_tests = len(os.listdir('./test_data'))
# test1 = np.loadtxt('./test_data/test_1.csv', delimiter=',')
# all_data = np.zeros((n_tests,) + test1.shape, dtype=float)
# for test in range(0, n_tests):
#     all_data[test] = np.loadtxt(f'./test_data/test_{test+1}.csv', delimiter=',')
# np.save('all_data.npy', all_data)
#%% for pinn tests
test0 = np.loadtxt('./data/pinn_data/test_0.csv', delimiter=',')
np.save('./data/pinn_data/test_0.npy', test0)
#%% with friction
n_tests = len(os.listdir('./data/with_friction'))
test1 = np.loadtxt('./data/with_friction/test_1.csv', delimiter=',')
all_data = np.zeros((n_tests,) + test1.shape, dtype=float)
for test in range(0, n_tests):
    all_data[test] = np.loadtxt(f'./data/with_friction/test_{test+1}.csv', delimiter=',')
np.save('./data/with_friction.npy', all_data)
#%% no friction
n_tests = len(os.listdir('./data/no_friction'))
test1 = np.loadtxt('./data/no_friction/test_1.csv', delimiter=',')
all_data = np.zeros((n_tests,) + test1.shape, dtype=float)
for test in range(0, n_tests):
    all_data[test] = np.loadtxt(f'./data/no_friction/test_{test+1}.csv', delimiter=',')
np.save('./data/no_friction.npy', all_data)