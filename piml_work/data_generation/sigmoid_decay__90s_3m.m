clc;
clear;
close;

e=exp(1); % e^x

starting_k = 1500;
stopping_k = 500;
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

% sigmoid parameters
t_offset=90;
beta_min=1;
beta_max=50;

%%
n_timesteps = floor(test_length/step_size);
t = linspace(0, test_length, n_timesteps)';
F = sin(2*pi*freq*t);
F_signal = [t, F];
%%
for i = 1:n_tests
    %%
    % produce k
    d=rand()*(beta_max-beta_min)+beta_min; % random constant

    k = 1000*1./(1+e.^(d*(t-90)))+stopping_k; % sigmoid function
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
        writematrix(arr, "./data/sigmoid_wf_90s_3m/test_"+i+".csv")
    else
        writematrix(arr, "./data/sigmoid_nf_90s_3m/test_"+i+".csv");
    end
end