clc;
clear;
close;

step_size = 0.001;
test_length = 600; % 10 minutes
m=1;
c=.2;
freq = 5;
n_tests = 100;
k_mid=1000;

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
%%
for i = 1:n_tests
    %%
    % produce k

    y_interp = [1 1];
    x_interp = [0, 1];
    y_interp = k_mid*y_interp;
    x_interp = x_interp*test_length;
    k = interp1(x_interp, y_interp, t);
    k_signal = [t, k];
    %%
    if with_friction
        out = sim('degrade_with_friction_v2.slx');
    else
        out = sim('degrade_no_friction_v2.slx');
    end
    save_data_out(out, i, with_friction);
    %%
    plot(k)
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
        writematrix(arr, "./data/k_mid_wf_10m/test_"+i+".csv")
    else
        writematrix(arr, "./data/k_mid_nf_10m/test_"+i+".csv");
    end
end