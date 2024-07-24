% 120 second tests, all with 5 Hz excitation
% The path of degradation will change for each test
% 100 tests total
starting_k = 1500;
stopping_k = 500;
step_size = 0.001;
test_length = 120;
m=1;
c=.2;
freq = 5;
n_tests = 100;

% friction element
F_brk = 0.02;
v_brk = 0.01;
F_c = 0.9*F_brk;

% time point range of first line segment end (as a ratio of test length)
t_range1 = [0.2, 0.4];
t_range2 = [0.6, 0.8];
% percentage of possible degredation at time point 1 and 2
k_range1 = [.55, .8];
k_range2 = [.2, .45];
%%
n_timesteps = floor(test_length/step_size);
t = linspace(0, test_length, n_timesteps)';
F = sin(freq*2*pi*t);
F_signal = [t, F];
%%
for i = 1:n_tests
    %%
    x1 = t_range1(1) + rand()*(t_range1(2)-t_range1(1));
    x2 = t_range2(1) + rand()*(t_range2(2)-t_range2(1));
    y1 = k_range1(1) + rand()*(k_range1(2)-k_range1(1));
    y2 = k_range2(1) + rand()*(k_range2(2)-k_range2(1));
    % produce k
    k = zeros(1, n_timesteps);
    for j = 1:n_timesteps
        x = t(j)/test_length;
        if x < x1
            z1 = 0; z2 = x1;
            w1 = 1; w2 = y1;
        elseif x < x2
            z1 = x1; z2 = x2;
            w1 = y1; w2 = y2;
        else
            z1 = x2; z2 = 1;
            w1 = y2; w2 = 0;
        end
        x = (x - z1)/(z2 - z1);
        x = w1 + x*(w2 - w1);
        k(j) = stopping_k + x*(starting_k - stopping_k);
    end
    k = k';
    k_signal = [t, k];
    %%
    out = sim('forced_degrade_v1.slx');
    save_data_out(out, i);
    %%
end
function save_data_out(out, i)
    x = out.displacement(:,2);
    v = out.velocity(:,2);
    a = out.acceleration(:,2);
    t = out.velocity(:,1);
    F = out.force(:,2);
    k = out.stiffness(:,2);

    arr = [t, x, v, a, k, F];
    writematrix(arr, "./test_data/test_"+i+".csv");
end