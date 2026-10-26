clc;
clear;
close;

starting_k = 1500;
stopping_k = 500;
step_size = 0.001;
test_length = 180;

same_mass=true;
same_damping=true;

if same_mass
    m=1;
    m1=m;
    m2 = m1;

else
    m1 = 1;
    m2 = 1.5;
end

if same_damping
    c=0.1;
    c1=c;
    c2=c;
    c3=c;
else
    c1=0.1;
    c2=0.2;
    c3=0.3;
end

freq = 5;
n_tests = 100;

with_friction = true;
% friction element
F_brk = 0.05;
v_brk = 0.01;
F_c = 0.9*F_brk;

% time point range of first line segment end (as a ratio of test length)
x_range1 = [0.35, 0.45];
x_range2 = [0.55, 0.65];
% percentage of possible degredation at time point 1 and 2
y_range1 = [.57, .85];
y_range2 = [.14, .43];
%%
n_timesteps = floor(test_length/step_size);
t = linspace(0, test_length, n_timesteps)';
F = sin(2*pi*freq*t);
F_signal = [t, F];

% same stiffness
k_same=false;
%%
for i = 1:n_tests
    %%
    % produce k
    if k_same
        x1 = x_range1(1) + rand()*(x_range1(2)-x_range1(1));
        x2 = x_range2(1) + rand()*(x_range2(2)-x_range2(1));
        y1 = y_range1(1) + rand()*(y_range1(2)-y_range1(1));
        y2 = y_range2(1) + rand()*(y_range2(2)-y_range2(1));
        y_interp = [1, 1, y1, y2, 0, 0];
        x_interp = [0, (1/3), x1, x2, (2/3), 1];
        y_interp = (starting_k - stopping_k)*y_interp + stopping_k;
        x_interp = x_interp*test_length;
        k1 = interp1(x_interp, y_interp, t);
        k2 = k1;
        k3 = k1;
    else
        x1_1 = x_range1(1) + rand()*(x_range1(2)-x_range1(1));
        x2_1 = x_range2(1) + rand()*(x_range2(2)-x_range2(1));
        y1_1 = y_range1(1) + rand()*(y_range1(2)-y_range1(1));
        y2_1 = y_range2(1) + rand()*(y_range2(2)-y_range2(1));

        y_interp_1 = [1, 1, y1_1, y2_1, 0, 0];
        x_interp_1 = [0, (1/3), x1_1, x2_1, (2/3), 1];
        y_interp_1 = (starting_k - stopping_k)*y_interp_1 + stopping_k;
        x_interp_1 = x_interp_1*test_length;
        k1 = interp1(x_interp_1, y_interp_1, t);

        x1_2 = x_range1(1) + rand()*(x_range1(2)-x_range1(1));
        x2_2 = x_range2(1) + rand()*(x_range2(2)-x_range2(1));
        y1_2 = y_range1(1) + rand()*(y_range1(2)-y_range1(1));
        y2_2 = y_range2(1) + rand()*(y_range2(2)-y_range2(1));

        y_interp_2 = [1, 1, y1_2, y2_2, 0, 0];
        x_interp_2 = [0, (1/3), x1_2, x2_2, (2/3), 1];
        y_interp_2 = (starting_k - stopping_k)*y_interp_2 + stopping_k;
        x_interp_2 = x_interp_2*test_length;
        k2 = interp1(x_interp_2, y_interp_2, t);

        x1_3 = x_range1(1) + rand()*(x_range1(2)-x_range1(1));
        x2_3 = x_range2(1) + rand()*(x_range2(2)-x_range2(1));
        y1_3 = y_range1(1) + rand()*(y_range1(2)-y_range1(1));
        y2_3 = y_range2(1) + rand()*(y_range2(2)-y_range2(1));

        y_interp_3 = [1, 1, y1_3, y2_3, 0, 0];
        x_interp_3 = [0, (1/3), x1_3, x2_3, (2/3), 1];
        y_interp_3 = (starting_k - stopping_k)*y_interp_3 + stopping_k;
        x_interp_3 = x_interp_3*test_length;
        k3 = interp1(x_interp_3, y_interp_3, t);
    end

    k_signal_1=[t, k1];
    k_signal_2=[t, k2];
    k_signal_3=[t, k3];
    
    figure(1)
    plot(t,k1)
    title('spring 1')
    xlabel('time (s)')
    ylabel('stiffness (N/m)')
    hold on

    figure(2)
    plot(t,k2)
    title('spring 2')
    xlabel('time (s)')
    ylabel('stiffness (N/m)')
    hold on

    figure(3)
    plot(t,k3)
    title('spring 3')
    xlabel('time (s)')
    ylabel('stiffness (N/m)')
    hold on
    
    %%
    if with_friction
        out = sim('degrade_with_friction_2dof.slx');
    else
        out = sim('degrade_no_friction_2dof.slx');
    end
    save_data_out(out, i, with_friction);
    %%
end
function save_data_out(out, i, with_friction)
    x1 = out.displacement_1(:,2);
    v1 = out.velocity_1(:,2);
    a1 = out.acceleration_1(:,2);
    t1 = out.velocity_1(:,1);
    x2 = out.displacement_2(:,2);
    v2 = out.velocity_2(:,2);
    a2 = out.acceleration_2(:,2);
    t2 = out.velocity_2(:,1);
    F = out.force(:,2);
    k1 = out.stiffness_1(:,2);
    k2 = out.stiffness_2(:,2);
    k3 = out.stiffness_3(:,2);
    if any(isnan(a1)) || any(isnan(a2))
        fprintf('nan values found.\n')
    end
    arr = [t1, x1, v1, a1, t2, x2, v2, a2, k1, k2, k3, F];
    if with_friction
        writematrix(arr, "./data/two_dof_decay_wf/test_"+i+".csv")
    else
        writematrix(arr, "./data/two_dof_decay_nf/test_"+i+".csv");
    end
end