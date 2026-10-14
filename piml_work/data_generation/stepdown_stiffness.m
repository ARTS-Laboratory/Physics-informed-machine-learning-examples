clc;
clear;
close;

starting_k = 1500;
stopping_k = 500;
step_size = 0.001;
test_length = 180; % 10 min test
m=1;
c=.2;
freq = 5;
n_tests = 100;
d=1e-5; % added to allow sharp rises in stiffness

with_friction = true;
% friction element
F_brk = 0.05;
v_brk = 0.01;
F_c = 0.9*F_brk;

% percentage of possible degredation at time point 1 and 2
y_range = [.1 .9];
%%
n_timesteps = floor(test_length/step_size);
t = linspace(0, test_length, n_timesteps)';
F = sin(2*pi*freq*t);
F_signal = [t, F];
%%
for i = 1:n_tests
    %%
    % produce k

    y0 = y_range(1) + rand()*(y_range(2)-y_range(1));

    y_interp = [1 1 y0 y0 0 0];
    x_interp = [0 (1/3-d) (1/3+d) (2/3-d) (2/3+d) 1];
    y_interp = (starting_k - stopping_k)*y_interp + stopping_k;
    x_interp = x_interp*test_length;
    k = interp1(x_interp, y_interp, t);
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
    plot(t,k)
    hold on
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
        writematrix(arr, "./data/stepdown_wf_3m/test_"+i+".csv")
    else
        writematrix(arr, "./data/stepdown_nf_3m/test_"+i+".csv");
    end
end