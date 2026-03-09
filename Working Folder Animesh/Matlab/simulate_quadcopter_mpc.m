function simulate_quadcopter_mpc()

clear;clc;close all

%% --- Parameters ---ßß
m = 1.0; g = 9.81;
Ixx = 0.01; Iyy = 0.01; Izz = 0.02;
kv = 0.1; kw = 0.01;
Ts = 0.01;

%% --- Build linearized model ---
[A, B, ~, sys_d] = quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts);
[Ad, Bd] = ssdata(sys_d);

%% --- MPC plant model ---
nx = 12; nu = 4;
C  = eye(nx);
D  = zeros(nx, nu);

plant = ss(Ad, Bd, C, D, Ts);
plant.StateName  = {'x','y','z','vx','vy','vz','phi','theta','psi','p','q','r'};
plant.InputName  = {'u1','u2','u3','u4'};
plant.OutputName = plant.StateName;

%% --- MPC object ---
N  = 10; %Prediction Horizon; how many steps ahed the MPC looks into the future when optimizing
Nc = 3;  % Control Horizon ; how many future control inputs the MPC is allowed to freely optimize

mpcobj = mpc(plant, Ts, N, Nc);

%% --- Nominal operating point ---
mpcobj.Model.Nominal.U  = [m*g; 0; 0; 0];
mpcobj.Model.Nominal.X  = zeros(nx, 1);
mpcobj.Model.Nominal.Y  = zeros(nx, 1);
mpcobj.Model.Nominal.DX = zeros(nx, 1);

%% --- Weights ---
mpcobj.Weights.OutputVariables           = [10 10 10  1  1  1  1  1  1  0.1 0.1 0.1];
mpcobj.Weights.ManipulatedVariables      = [0.1 0.1 0.1 0.1];
mpcobj.Weights.ManipulatedVariablesRate  = [0.05 0.05 0.05 0.05];

%% --- State constraints ---
mpcobj.OutputVariables(1).Min = -5;   mpcobj.OutputVariables(1).Max =  5;
mpcobj.OutputVariables(2).Min = -5;   mpcobj.OutputVariables(2).Max =  5;
mpcobj.OutputVariables(3).Min = -0.5; mpcobj.OutputVariables(3).Max =  5;
mpcobj.OutputVariables(4).Min = -3;   mpcobj.OutputVariables(4).Max =  3;
mpcobj.OutputVariables(5).Min = -3;   mpcobj.OutputVariables(5).Max =  3;
mpcobj.OutputVariables(6).Min = -3;   mpcobj.OutputVariables(6).Max =  3;
mpcobj.OutputVariables(7).Min  = -0.3; mpcobj.OutputVariables(7).Max  = 0.3;
mpcobj.OutputVariables(8).Min  = -0.3; mpcobj.OutputVariables(8).Max  = 0.3;
mpcobj.OutputVariables(9).Min  = -pi;  mpcobj.OutputVariables(9).Max  = pi;
mpcobj.OutputVariables(10).Min = -2;  mpcobj.OutputVariables(10).Max =  2;
mpcobj.OutputVariables(11).Min = -2;  mpcobj.OutputVariables(11).Max =  2;
mpcobj.OutputVariables(12).Min = -2;  mpcobj.OutputVariables(12).Max =  2;

%% --- Reference ---
x_ref      = zeros(1, nx);
x_ref(1:3) = [1, 1, 1];

%% --- Simulation ---
t_end = 10;
t     = 0:Ts:t_end;
nstep = length(t);

% MPC closed-loop trajectory (plant model inside MPC)
X_mpc    = zeros(nx, nstep);

% Actual linear model trajectory (propagated with quadcopter_dynamics)
X_actual = zeros(nx, nstep);
U_actual = zeros(nu, nstep);

x_mpc    = zeros(nx, 1);
x_actual = zeros(nx, 1);
xmpc     = mpcstate(mpcobj);

for k = 1:nstep
    X_mpc(:,k)    = x_mpc;
    X_actual(:,k) = x_actual;

    % MPC computes u based on its internal plant model
    u             = mpcmove(mpcobj, xmpc, x_mpc, x_ref);
    U_actual(:,k) = u;

    % MPC internal model step (what MPC thinks will happen)
    x_mpc = Ad * x_mpc + Bd * (u - [m*g;0;0;0]);

    % Actual linear model step (ground truth propagation)
    x_dot    = quadcopter_dynamics(x_actual, u, A, B, m, g);
    x_actual = x_actual + Ts * x_dot;
end

%% --- Plot: MPC predicted vs Actual (position only) ---
state_names = {'x','y','z','vx','vy','vz','phi','theta','psi','p','q','r'};
ref_vals    = [1 1 1 0 0 0 0 0 0 0 0 0];

figure('Name','MPC vs Actual — Position');
pos_labels = {'x [m]','y [m]','z [m]'};
for i = 1:3
    subplot(3,1,i);
    plot(t, X_mpc(i,:),    'b',  'LineWidth', 1.5); hold on;
    plot(t, X_actual(i,:), 'r--','LineWidth', 1.5);
    yline(ref_vals(i), 'g:', 'LineWidth', 1.2);
    grid on;
    xlabel('t [s]'); ylabel(pos_labels{i});
    title(pos_labels{i});
    legend('MPC internal','Actual linear model','Reference','Location','best');
end
sgtitle('Linear MPC vs Actual Linear Model — Position Tracking');

%% --- Plot all 12 states ---
figure('Name','MPC vs Actual — All States');
for i = 1:12
    subplot(4,3,i);
    plot(t, X_mpc(i,:),    'b',  'LineWidth', 1.2); hold on;
    plot(t, X_actual(i,:), 'r--','LineWidth', 1.2);
    yline(ref_vals(i), 'g:', 'LineWidth', 1.0);
    grid on;
    xlabel('t [s]'); ylabel(state_names{i});
    title(state_names{i});
end
sgtitle('Linear MPC vs Actual Linear Model — All States');
legend('MPC internal','Actual','Reference','Location','best');

%% --- Plot control inputs ---
input_names = {'u1 (thrust)','u2 (roll)','u3 (pitch)','u4 (yaw)'};
figure('Name','Control Inputs');
for i = 1:4
    subplot(2,2,i);
    plot(t, U_actual(i,:), 'g', 'LineWidth', 1.2); grid on;
    xlabel('t [s]'); ylabel(input_names{i});
    title(input_names{i});
end
sgtitle('Quadcopter Linear MPC — Control Inputs');

%% --- 3D Trajectory: MPC vs Actual ---
figure('Name','3D Trajectory MPC vs Actual');
plot3(X_mpc(1,:),    X_mpc(2,:),    X_mpc(3,:),    'b',  'LineWidth', 1.5); hold on;
plot3(X_actual(1,:), X_actual(2,:), X_actual(3,:), 'r--','LineWidth', 1.5);
plot3(0, 0, 0, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(1, 1, 1, 'r*', 'MarkerSize', 12);
grid on; axis equal;
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
title('Quadcopter MPC — 3D Trajectory');
legend('MPC internal','Actual linear model','Start','Reference [1,1,1]','Location','best');
view(45, 30);

end