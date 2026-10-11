clc;
clear;
close;

starting_k = 1500;
stopping_k = 500;
k_mid=starting_k-stopping_k;
step_size = 0.001;
test_length = 180;
m=1;
c=.2;
freq = 5;
n_tests = 100;

with_friction = true;
% friction element
F_brk = 0.05;
v_brk = 0.01;
F_c = 0.9*F_brk;

%%
n_timesteps = floor(test_length/step_size);
t = linspace(0, test_length, n_timesteps)';
F = sin(2*pi*freq*t);
F_signal = [t, F];

% parameters for Mathieu system
omega=pi/(0.5*test_length);

%%
for i = 1:n_tests
    %%
    % produce k
    k_amp=rand()*.5;
    k=k_mid*(1+k_amp*cos(omega.*t));
    k_signal = [t, k];

    plot(t,k)
    hold on
    %%
    if with_friction
        out = sim('degrade_with_friction_v2.slx');
    else
        out = sim('degrade_no_friction_v2.slx');
    end
    save_data_out(out, i, with_friction);
    %%
end
function save_data_out(out, i, with_friction)
    x = out.displacement(:,2);
    v = out.velocity(:,2);
    a = out.acceleration(:,2);
    t = out.velocity(:,1);
    F = out.force(:,2);
    k = out.stiffness(:,2);
    if any(isnan(a))
        fprintf('nan values found.')
    end
    arr = [t, x, v, a, k, F];
    if with_friction
        writematrix(arr, "./data/mathieu_k_mean_one_cycle_wf/test_"+i+".csv")
    else
        writematrix(arr, "./data/mathieu_k_mean_one_cycle_nf/test_"+i+".csv");
    end
end