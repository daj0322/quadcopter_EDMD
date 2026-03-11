function simulate_leader_follower()
clear;clc;close all
%% --- Parameters ---
m = 1.0; g = 9.81;
Ixx = 0.01; Iyy = 0.01; Izz = 0.02;
kv = 0.1; kw = 0.01;
Ts = 0.01;

%% --- Build model ---
[A, B, ~, sys_d] = quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts);
[Ad, Bd] = ssdata(sys_d);

%% --- PD Gains (leader) ---
Kp = zeros(4,12);
Kd = zeros(4,12);
Kp(1,3)  =  2.0;  Kd(1,6)  =  1.5;
Kp(2,2)  = -0.5;  Kd(2,5)  = -0.3;
Kp(2,7)  =  3.0;  Kd(2,10) =  0.5;
Kp(3,1)  =  0.5;  Kd(3,4)  =  0.3;
Kp(3,8)  =  3.0;  Kd(3,11) =  0.5;
Kp(4,9)  =  1.0;  Kd(4,12) =  0.3;

%% --- MPC setup (follower) ---
nx = 12; nu = 4;
C  = eye(nx);
D  = zeros(nx, nu);

plant = ss(Ad, Bd, C, D, Ts);
plant.StateName  = {'x','y','z','vx','vy','vz','phi','theta','psi','p','q','r'};
plant.InputName  = {'u1','u2','u3','u4'};
plant.OutputName = plant.StateName;

N  = 10;
Nc = 3;
mpcobj = mpc(plant, Ts, N, Nc);

mpcobj.Model.Nominal.U  = [m*g; 0; 0; 0];
mpcobj.Model.Nominal.X  = zeros(nx, 1);
mpcobj.Model.Nominal.Y  = zeros(nx, 1);
mpcobj.Model.Nominal.DX = zeros(nx, 1);

mpcobj.Weights.OutputVariables          = [10 10 10  1  1  1  1  1  1  0.1 0.1 0.1];
mpcobj.Weights.ManipulatedVariables     = [0.1 0.1 0.1 0.1];
mpcobj.Weights.ManipulatedVariablesRate = [0.05 0.05 0.05 0.05];

mpcobj.OutputVariables(1).Min = -5;    mpcobj.OutputVariables(1).Max =  5;
mpcobj.OutputVariables(2).Min = -5;    mpcobj.OutputVariables(2).Max =  5;
mpcobj.OutputVariables(3).Min = -5;    mpcobj.OutputVariables(3).Max =  5;
mpcobj.OutputVariables(4).Min = -3;    mpcobj.OutputVariables(4).Max =  3;
mpcobj.OutputVariables(5).Min = -3;    mpcobj.OutputVariables(5).Max =  3;
mpcobj.OutputVariables(6).Min = -3;    mpcobj.OutputVariables(6).Max =  3;
mpcobj.OutputVariables(7).Min  = -0.3; mpcobj.OutputVariables(7).Max  = 0.3;
mpcobj.OutputVariables(8).Min  = -0.3; mpcobj.OutputVariables(8).Max  = 0.3;
mpcobj.OutputVariables(9).Min  = -pi;  mpcobj.OutputVariables(9).Max  = pi;
mpcobj.OutputVariables(10).Min = -2;   mpcobj.OutputVariables(10).Max =  2;
mpcobj.OutputVariables(11).Min = -2;   mpcobj.OutputVariables(11).Max =  2;
mpcobj.OutputVariables(12).Min = -2;   mpcobj.OutputVariables(12).Max =  2;

%% --- Initial states ---
x_leader        = zeros(nx, 1);
x_leader(1:3)   = [0; 0; 0];    % leader starts at [0,0,0]

x_follower       = zeros(nx, 1);
x_follower(1:3)  = [1; 0; 0];   % follower starts at [1,0,0]

% Leader target
leader_ref       = zeros(1, nx);
leader_ref(1:3)  = [1, 1, 1];

%% --- Pre-simulate leader trajectory to find interception point ---
t_end  = 15;
t      = 0:Ts:t_end;
nstep  = length(t);

X_leader_full = zeros(nx, nstep);
x_sim         = x_leader;
for k = 1:nstep
    X_leader_full(:,k) = x_sim;
    u_sim  = quadcopter_pd_controller(x_sim, leader_ref', Kp, Kd, m, g);
    x_dot  = quadcopter_dynamics(x_sim, u_sim, A, B, m, g);
    x_sim  = x_sim + Ts * x_dot;
end

%% --- Find interception point ---
max_follower_speed = 1.5;   % [m/s] approximate max speed under MPC
intercept_k        = nstep; % default to end
for k = 1:nstep
    dist        = norm(X_leader_full(1:3, k) - x_follower(1:3));
    time_needed = dist / max_follower_speed;
    time_avail  = (k-1) * Ts;
    if time_avail >= time_needed
        intercept_k = k;
        break;
    end
end

intercept_pos    = X_leader_full(1:3, intercept_k);
intercept_plot_k = intercept_k;
intercept_t      = (intercept_k - 1) * Ts;
fprintf('Interception point : [%.2f, %.2f, %.2f]\n', ...
         intercept_pos(1), intercept_pos(2), intercept_pos(3));
fprintf('Interception time  : %.2f s\n', intercept_t);

%% --- Main simulation loop ---
X_leader   = zeros(nx, nstep);
X_follower = zeros(nx, nstep);
U_leader   = zeros(nu, nstep);
U_follower = zeros(nu, nstep);

x_leader(1:3)  = [0; 0; 0];   % reset leader
x_follower(1:3) = [1; 0; 0];  % reset follower
xmpc           = mpcstate(mpcobj);
intercepted    = false;

for k = 1:nstep
    X_leader(:,k)   = x_leader;
    X_follower(:,k) = x_follower;

    %% Leader: PD flies to [1,1,1]
    u_leader        = quadcopter_pd_controller(x_leader, leader_ref', Kp, Kd, m, g);
    U_leader(:,k)   = u_leader;
    x_dot_leader    = quadcopter_dynamics(x_leader, u_leader, A, B, m, g);
    x_leader        = x_leader + Ts * x_dot_leader;

    %% Follower: MPC aims at interception point, then locks onto leader
    if norm(x_follower(1:3) - x_leader(1:3)) < 0.05
        intercepted = true;
    end

    if ~intercepted
        follower_ref       = zeros(1, nx);
        follower_ref(1:3)  = intercept_pos';   % aim at interception point
    else
        follower_ref       = x_leader';        % lock onto leader after intercept
    end

    u_follower        = mpcmove(mpcobj, xmpc, x_follower, follower_ref);
    U_follower(:,k)   = u_follower;
    x_dot_follower    = quadcopter_dynamics(x_follower, u_follower, A, B, m, g);
    x_follower        = x_follower + Ts * x_dot_follower;
end

%% --- Plot 1: Full 3D trajectory (both drones full path) ---
figure('Name','3D Full Trajectories');

plot3(X_leader(1,:),   X_leader(2,:),   X_leader(3,:),   'b',  'LineWidth', 1.5); hold on;
plot3(X_follower(1,:), X_follower(2,:), X_follower(3,:), 'r--','LineWidth', 1.5);
plot3(0, 0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot3(1, 0, 0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot3(intercept_pos(1), intercept_pos(2), intercept_pos(3), 'k*', 'MarkerSize', 14);
plot3(1, 1, 1, 'g*', 'MarkerSize', 12);
grid on; axis equal;
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
title('3D Full Trajectories');
legend('Leader (PD)','Follower (MPC)','Leader start','Follower start', ...
       'Intercept point','Leader target','Location','best');
view(45, 30);
ax = gca;
ax.Color = 'w';                  % axes background
ax.GridColor = [0 0 0];          % black grid lines
ax.GridAlpha = 1;                % fully opaque grid
ax.MinorGridColor = [0 0 0];
ax.XColor = 'w';                 % black axis lines & tick labels
ax.YColor = 'w';
ax.ZColor = 'w';

%% --- Plot 2: 3D trajectory — leader clipped at interception ---
figure('Name','3D Intercept — Leader clipped');
plot3(X_leader(1, 1:intercept_plot_k), ...
      X_leader(2, 1:intercept_plot_k), ...
      X_leader(3, 1:intercept_plot_k), ...
      'b', 'LineWidth', 1.5); hold on;
plot3(X_follower(1,:), X_follower(2,:), X_follower(3,:), 'r--','LineWidth', 1.5);
plot3(0, 0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot3(1, 0, 0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot3(X_leader(1, intercept_plot_k), ...
      X_leader(2, intercept_plot_k), ...
      X_leader(3, intercept_plot_k), 'k*', 'MarkerSize', 14);
grid on; axis equal;
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
title('3D Intercept — Leader hidden after interception');
legend('Leader (clipped)','Follower (MPC)','Leader start', ...
       'Follower start','Intercept point','Location','best');
view(45, 30);

ax = gca;
ax.Color = 'w';                  % axes background
ax.GridColor = [0 0 0];          % black grid lines
ax.GridAlpha = 1;                % fully opaque grid
ax.MinorGridColor = [0 0 0];
ax.XColor = 'w';                 % black axis lines & tick labels
ax.YColor = 'w';
ax.ZColor = 'w';
%% --- Plot 3: Position over time — leader clipped at interception ---
pos_labels = {'x [m]','y [m]','z [m]'};
figure('Name','Position — Leader clipped at interception');
for i = 1:3
    subplot(3,1,i);
    plot(t(1:intercept_plot_k), X_leader(i, 1:intercept_plot_k), ...
         'b', 'LineWidth', 1.5); hold on;
    plot(t, X_follower(i,:), 'r--', 'LineWidth', 1.5);
    xline(intercept_t, 'k--', 'Intercept', 'LineWidth', 1.0);
    grid on;
    xlabel('t [s]'); ylabel(pos_labels{i});
    title(pos_labels{i});
    legend('Leader (clipped)','Follower (MPC)','Intercept','Location','best');

ax = gca;
ax.Color = 'w';                  % axes background
ax.GridColor = [0 0 0];          % black grid lines
ax.GridAlpha = 1;                % fully opaque grid
ax.MinorGridColor = [0 0 0];
ax.XColor = 'w';                 % black axis lines & tick labels
ax.YColor = 'w';
ax.ZColor = 'w';
end
sgtitle('Position — Leader clipped at interception');


%% --- Plot 4: Separation distance ---
dist_vec = vecnorm(X_leader(1:3,:) - X_follower(1:3,:), 2, 1);
figure('Name','Separation Distance');
plot(t, dist_vec, 'g', 'LineWidth', 1.5); grid on;
xlabel('t [s]'); ylabel('Distance [m]');
title('Separation Distance Between Leader and Follower');
yline(0.05, 'r--', 'Intercept threshold (0.05m)', 'LineWidth', 1.0);
xline(intercept_t, 'b--', sprintf('t = %.2fs', intercept_t), 'LineWidth', 1.0);
ax = gca;
ax.Color = 'w';                  % axes background
ax.GridColor = [0 0 0];          % black grid lines
ax.GridAlpha = 1;                % fully opaque grid
ax.MinorGridColor = [0 0 0];
ax.XColor = 'w';                 % black axis lines & tick labels
ax.YColor = 'w';
ax.ZColor = 'w';

%% --- Plot 5: Control inputs ---
input_names = {'u1 (thrust)','u2 (roll)','u3 (pitch)','u4 (yaw)'};
figure('Name','Control Inputs');
for i = 1:4
    subplot(2,2,i);
    plot(t, U_leader(i,:),   'b',  'LineWidth', 1.2); hold on;
    plot(t, U_follower(i,:), 'r--','LineWidth', 1.2);
    grid on;
    xlabel('t [s]'); ylabel(input_names{i});
    title(input_names{i});
    legend('Leader','Follower','Location','best');
ax = gca;
ax.Color = 'w';                  % axes background
ax.GridColor = [0 0 0];          % black grid lines
ax.GridAlpha = 1;                % fully opaque grid
ax.MinorGridColor = [0 0 0];
ax.XColor = 'w';                 % black axis lines & tick labels
ax.YColor = 'w';
ax.ZColor = 'w';
end
sgtitle('Control Inputs — Leader vs Follower');

end