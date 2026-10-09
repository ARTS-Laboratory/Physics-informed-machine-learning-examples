clc;
clear;
close;

starting_k = 1500;
stopping_k = 500;
step_size = 0.001;
test_length = 600; % 10 min test
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

% time point range of first line segment end (as a ratio of test length)

% percentage of possible degredation at time point 1 and 2
y_range1 = [.95 .99];
y_range2 = [.8 .9];
y_range3 = [.65 .75];
y_range4 = [.5 .6];
y_range5 = [.35 .45];
y_range6 = [.2 .3];
y_range7 = [0.05 .15];
y_range8 = [0.01 .04];
%%
n_timesteps = floor(test_length/step_size);
t = linspace(0, test_length, n_timesteps)';
F = sin(2*pi*freq*t);
F_signal = [t, F];
%%
for i = 1:n_tests
    %%
    % produce k

    y1 = y_range8(1) + rand()*(y_range8(2)-y_range8(1));
    y2 = y_range7(1) + rand()*(y_range7(2)-y_range7(1));
    y3 = y_range6(1) + rand()*(y_range6(2)-y_range6(1));
    y4 = y_range5(1) + rand()*(y_range5(2)-y_range5(1));
    y5 = y_range4(1) + rand()*(y_range4(2)-y_range4(1));
    y6 = y_range3(1) + rand()*(y_range3(2)-y_range3(1));
    y7 = y_range2(1) + rand()*(y_range2(2)-y_range2(1));
    y8 = y_range1(1) + rand()*(y_range1(2)-y_range1(1));


    y_interp = [0 0 y1 y1 y2 y2 y3 y3 y4 y4 y5 y5 y6 y6 y7 y7 y8 y8 1 1];
    x_interp = [0 (.1-d) (.1+d) (.2-d) (.2+d) (.3-d) (.3+d) (.4-d) (.4+d)...
        (.5-d) (.5+d) (.6-d) (.6+d) (.7-d) (.7+d) (.8-d) (.8+d) (.9-d) (.9+d) 1];
    y_interp = (starting_k - stopping_k)*y_interp + stopping_k;
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
        writematrix(arr, "./data/stepup_wf_10m/test_"+i+".csv")
    else
        writematrix(arr, "./data/stepup_nf_10m/test_"+i+".csv");
    end
end