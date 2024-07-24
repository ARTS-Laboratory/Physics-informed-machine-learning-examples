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
test0 = np.loadtxt('./pinn_data/test_0.csv', delimiter=',')
np.save('./pinn_data/test_0.npy', test0)
#%% with friction
n_tests = len(os.listdir('./with_friction'))
test1 = np.loadtxt('./with_friction/test_1.csv', delimiter=',')
all_data = np.zeros((n_tests,) + test1.shape, dtype=float)
for test in range(0, n_tests):
    all_data[test] = np.loadtxt(f'./with_friction/test_{test+1}.csv', delimiter=',')
np.save('with_friction.npy', all_data)
#%% no friction
n_tests = len(os.listdir('./no_friction'))
test1 = np.loadtxt('./no_friction/test_1.csv', delimiter=',')
all_data = np.zeros((n_tests,) + test1.shape, dtype=float)
for test in range(0, n_tests):
    all_data[test] = np.loadtxt(f'./no_friction/test_{test+1}.csv', delimiter=',')
np.save('no_friction.npy', all_data)