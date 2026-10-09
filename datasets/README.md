This folder contains the MATLAB code and .csv files for each of the datasets. This is meant to demonstrate how the PIML models operate under varying conditions. Each code was run using the given Simulink models ("degrade_no_friction_v2.slx" and "degrade_with_friction_v2.slx)

# Current Datasets
1) run_experiments_decaying_2m.m (original MATLAB code; 120s)
2) run_experiments_decaying_3m.m (3 minutes; one minute of constant stiffness, 1 minute of decay, 1 minute of constant stiffness)
3) run_experiments_k_mid_10m.m (stiffness is at 1000 N/m for each test for 10 minutes)
4) run_experiments_rising_k.m (stiffness rises similar to code 1)
5) run_experiments_constant_k_10.m (stiffness is constant for the entire 10 min test; stiffness value is different for each test)
6) run_experiments_stepdown_10m.m (stiffness drops every minute)
7) run_experiments_stepup_10m.m (stiffness sharply increases every minute)
