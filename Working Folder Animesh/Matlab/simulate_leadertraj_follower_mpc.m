function simulate_leadertraj_follower_mpc()
% Simulation: Reference Trajectory (helical) + Follower drone (MPC- Linear)
%
% Reference : A pre-computed multi-segment helical path — NOT a simulated
%             drone. It is a purely geometric trajectory evaluated at each
%             timestep. No dynamics, no controller on the reference side.
%
%   Segment 1 (0   → T1): Lift-off straight up from [R,0,0] to helix base
%   Segment 2 (T1  → T2): Helix rising — radius 2 m, rise 4 m, one full loop
%   Segment 3 (T2  → T3): Horizontal circle at peak altitude (half loop)
%   Segment 4 (T3  → T4): Helix descending — half loop, drop 2 m
%   Total duration ≈ 12 s  (medium speed)
%
% Follower : MPC tracking the LIVE reference position at each timestep.
%            Two trajectories recorded:
%              X_follower_mpc    – MPC internal model state
%              X_follower_actual – ground-truth linear dynamics
%
% Interception:
%   Capture radius R_capture = 0.05 m.
%   Follower must stay inside for T_dwell = 2 s continuously → confirmed.
%   Simulation ends immediately on confirmation.
%
% Plots (clipped to interception + 0.5 s buffer):
%   1. 3D — full reference path + follower trajectories
%   2. 3D — reference clipped at confirmed interception
%   3. Position x/y/z over time
%   4. Separation distance vs time
%   5. Follower control inputs
%   6. All 12 follower states — MPC internal vs actual

clear; clc; close all;

%% -----------------------------------------------------------------------
%  Physical parameters
% ------------------------------------------------------------------------
m   = 1.0;  g   = 9.81;
Ixx = 0.01; Iyy = 0.01; Izz = 0.02;
kv  = 0.1;  kw  = 0.01;
Ts  = 0.01;

%% -----------------------------------------------------------------------
%  Capture / dwell / plot parameters
% ------------------------------------------------------------------------
R_capture = 0.05;                    % capture radius [m]
T_dwell   = 2.0;                     % required dwell time [s]
N_dwell   = round(T_dwell / Ts);     % dwell steps (= 200)
plot_buf  = 0.5;                     % post-interception buffer [s]

%% -----------------------------------------------------------------------
%  Linearised quadcopter model  (follower plant)
% ------------------------------------------------------------------------
[A, B, ~, sys_d] = quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts);
[Ad, Bd]         = ssdata(sys_d);

%% -----------------------------------------------------------------------
%  MPC setup (follower)
% ------------------------------------------------------------------------
nx = 12; nu = 4;
plant = ss(Ad, Bd, eye(nx), zeros(nx,nu), Ts);
plant.StateName  = {'x','y','z','vx','vy','vz','phi','theta','psi','p','q','r'};
plant.InputName  = {'u1','u2','u3','u4'};
plant.OutputName = plant.StateName;

N  = 30;    % prediction horizon
Nc = 3;     % control horizon
mpcobj = mpc(plant, Ts, N, Nc);

mpcobj.Model.Nominal.U  = [m*g; 0; 0; 0];
mpcobj.Model.Nominal.X  = zeros(nx, 1);
mpcobj.Model.Nominal.Y  = zeros(nx, 1);
mpcobj.Model.Nominal.DX = zeros(nx, 1);

% Weights — heavy on position, light on angular rates
mpcobj.Weights.OutputVariables          = [10 10 10  1  1  1  1  1  1  0.1 0.1 0.1];
mpcobj.Weights.ManipulatedVariables     = [0.1 0.1 0.1 0.1];
mpcobj.Weights.ManipulatedVariablesRate = [0.05 0.05 0.05 0.05];

% State / output constraints (z upper limit raised for helix altitude)
lims = [ -5  5;  -5  5;  -0.5  6;  -3  3;  -3  3;  -3  3; ...
         -0.3  0.3;  -0.3  0.3;  -pi  pi;  -2  2;  -2  2;  -2  2];
for i = 1:nx
    mpcobj.OutputVariables(i).Min = lims(i,1);
    mpcobj.OutputVariables(i).Max = lims(i,2);
end

%% -----------------------------------------------------------------------
%  Reference trajectory — multi-segment helical path
%  Evaluated analytically; NO drone dynamics involved on the reference side.
% ------------------------------------------------------------------------
%
%  Helix geometry
%    Centre      : [0, 0]  in x-y
%    Radius      : R_helix = 2 m
%    Total rise  : H_helix = 4 m  (Segment 2)
%    Lift-off alt: z_base  = 0.5 m
%
%  Segment timing
%    Seg 1   0   →  1 s   straight lift-off
%    Seg 2   1   →  6 s   rising helix, 1 full turn  (5 s → medium speed)
%    Seg 3   6   →  9 s   flat circle at peak, half turn
%    Seg 4   9   → 12 s   descending helix, half turn, -2 m

t_end  = 15;
t_full = 0:Ts:t_end;
nstep  = length(t_full);

R_helix = 2.0;
H_helix = 4.0;
H_down  = 2.0;
z_base  = 0.5;

T1 = 1.0;
T2 = T1 + 5.0;   % = 6 s
T3 = T2 + 3.0;   % = 9 s
T4 = T3 + 3.0;   % = 12 s

% Pre-compute reference positions
ref_pos = zeros(3, nstep);

% Index of the last trajectory step (for hold after T4)
k_T4 = min(round(T4/Ts) + 1, nstep);

for k = 1:nstep
    tk = t_full(k);

    if tk <= T1
        % Segment 1: vertical lift-off, x=R_helix, y=0
        alpha        = tk / T1;
        ref_pos(:,k) = [R_helix; 0; alpha * z_base];

    elseif tk <= T2
        % Segment 2: rising helix, 1 full turn
        alpha        = (tk - T1) / (T2 - T1);
        theta_h      = 2*pi * alpha;
        ref_pos(:,k) = [ R_helix * cos(theta_h); ...
                         R_helix * sin(theta_h); ...
                         z_base + H_helix * alpha ];

    elseif tk <= T3
        % Segment 3: flat circle at peak altitude, half turn
        z_peak       = z_base + H_helix;
        alpha        = (tk - T2) / (T3 - T2);
        theta_h      = 2*pi + pi * alpha;   % continue winding: 2pi → 3pi
        ref_pos(:,k) = [ R_helix * cos(theta_h); ...
                         R_helix * sin(theta_h); ...
                         z_peak ];

    elseif tk <= T4
        % Segment 4: descending helix, half turn
        z_peak       = z_base + H_helix;
        alpha        = (tk - T3) / (T4 - T3);
        theta_h      = 3*pi + pi * alpha;   % continue: 3pi → 4pi
        ref_pos(:,k) = [ R_helix * cos(theta_h); ...
                         R_helix * sin(theta_h); ...
                         z_peak - H_down * alpha ];

    else
        % Hold final position
        ref_pos(:,k) = ref_pos(:, k_T4);
    end
end

fprintf('Reference trajectory: 4 segments, total active path = %.1f s\n', T4);

%% -----------------------------------------------------------------------
%  Initial follower state — offset so interception is non-trivial
% ------------------------------------------------------------------------
x_follower      = zeros(nx, 1);
x_follower(1:3) = [0; -1; 0];   % 1 m to the side of the ref start

%% -----------------------------------------------------------------------
%  Storage arrays
% ------------------------------------------------------------------------
X_ref             = zeros(3,  nstep);
X_follower_mpc    = zeros(nx, nstep);
X_follower_actual = zeros(nx, nstep);
U_follower        = zeros(nu, nstep);

%% -----------------------------------------------------------------------
%  MPC state object
% ------------------------------------------------------------------------
x_mpc_internal = x_follower;
xmpc           = mpcstate(mpcobj);

%% -----------------------------------------------------------------------
%  Capture / dwell counters
% ------------------------------------------------------------------------
in_radius     = false;
dwell_count   = 0;
capture_k     = nstep;   % sentinel
confirmed_k   = nstep;   % sentinel
sim_confirmed = false;

fprintf('Running main simulation loop...\n');

%% -----------------------------------------------------------------------
%  Main simulation loop
% ------------------------------------------------------------------------
for k = 1:nstep

    % Record states
    X_ref(:,k)             = ref_pos(:,k);
    X_follower_mpc(:,k)    = x_mpc_internal;
    X_follower_actual(:,k) = x_follower;

    % ----------------------------------------------------------------
    %  Capture radius / dwell check
    % ----------------------------------------------------------------
    sep = norm(x_follower(1:3) - ref_pos(:,k));

    if sep <= R_capture
        if ~in_radius
            in_radius   = true;
            capture_k   = k;
            dwell_count = 1;
        else
            dwell_count = dwell_count + 1;
        end

        if dwell_count >= N_dwell
            confirmed_k   = k;
            sim_confirmed = true;
            fprintf('INTERCEPTION CONFIRMED at t = %.2f s  (%.1f s inside capture radius)\n', ...
                    (k-1)*Ts, T_dwell);
            break;
        end
    else
        in_radius   = false;
        dwell_count = 0;
    end

    % ----------------------------------------------------------------
    %  Follower MPC reference = live reference trajectory position
    % ----------------------------------------------------------------
    follower_ref      = zeros(1, nx);
    follower_ref(1:3) = ref_pos(:,k)';

    % ----------------------------------------------------------------
    %  MPC solve
    % ----------------------------------------------------------------
    u_follower       = mpcmove(mpcobj, xmpc, x_mpc_internal, follower_ref);
    U_follower(:,k)  = u_follower;

    % MPC internal model update
    x_mpc_internal = Ad * x_mpc_internal + Bd * (u_follower - [m*g;0;0;0]);

    % Actual follower dynamics (ground-truth)
    x_dot_follower = quadcopter_dynamics(x_follower, u_follower, A, B, m, g);
    x_follower     = x_follower + Ts * x_dot_follower;
end

%% -----------------------------------------------------------------------
%  Post-loop: determine plot window
% ------------------------------------------------------------------------
if sim_confirmed
    sim_end_k = confirmed_k;
else
    sim_end_k = nstep;
    warning('Interception NOT confirmed within simulation window.');
end

% Safety clamp for capture_k sentinel
if ~sim_confirmed || capture_k >= sim_end_k
    capture_k = sim_end_k;
end

t_confirmed     = (sim_end_k - 1) * Ts;
t_capture_start = (capture_k  - 1) * Ts;

buf_steps  = min(round(plot_buf / Ts), nstep - sim_end_k);
plot_end_k = sim_end_k + buf_steps;
t_plot     = t_full(1:plot_end_k);

fprintf('Confirmed interception time : %.2f s\n', t_confirmed);
fprintf('Dwell began at              : %.2f s\n', t_capture_start);
fprintf('Plot window ends at         : %.2f s\n', t_full(plot_end_k));

%% -----------------------------------------------------------------------
%  Plot 1 — 3D full view
% ------------------------------------------------------------------------
figure('Name','3D — Reference path & Follower');
% Ghost of full reference path
plot3(ref_pos(1,:), ref_pos(2,:), ref_pos(3,:), ...
      'b:','LineWidth',1.0,'DisplayName','Reference path (full)'); hold on;
% Reference up to plot window
plot3(X_ref(1,1:plot_end_k), X_ref(2,1:plot_end_k), X_ref(3,1:plot_end_k), ...
      'b','LineWidth',2.0,'DisplayName','Reference (active)');
plot3(X_follower_mpc(1,1:plot_end_k), X_follower_mpc(2,1:plot_end_k), ...
      X_follower_mpc(3,1:plot_end_k), ...
      'r--','LineWidth',1.5,'DisplayName','Follower MPC internal');
plot3(X_follower_actual(1,1:plot_end_k), X_follower_actual(2,1:plot_end_k), ...
      X_follower_actual(3,1:plot_end_k), ...
      'm:','LineWidth',1.8,'DisplayName','Follower actual');
% Start markers
plot3(ref_pos(1,1), ref_pos(2,1), ref_pos(3,1), ...
      'bs','MarkerSize',10,'MarkerFaceColor','b','DisplayName','Ref start');
plot3(X_follower_actual(1,1), X_follower_actual(2,1), X_follower_actual(3,1), ...
      'ro','MarkerSize',10,'MarkerFaceColor','r','DisplayName','Follower start');
% Capture entry and confirmed markers
plot3(X_follower_actual(1,capture_k), X_follower_actual(2,capture_k), ...
      X_follower_actual(3,capture_k), ...
      'gs','MarkerSize',12,'LineWidth',2, ...
      'DisplayName',sprintf('Entered capture r (t=%.2fs)', t_capture_start));
plot3(X_follower_actual(1,sim_end_k), X_follower_actual(2,sim_end_k), ...
      X_follower_actual(3,sim_end_k), ...
      'g^','MarkerSize',14,'LineWidth',2,'MarkerFaceColor','g', ...
      'DisplayName',sprintf('Confirmed (t=%.2fs)', t_confirmed));
grid on; axis equal;
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
title(sprintf('3D — Interception confirmed at t = %.2f s', t_confirmed));
legend('Location','best');
view(45,30);

%% -----------------------------------------------------------------------
%  Plot 2 — 3D: reference clipped at confirmed interception
% ------------------------------------------------------------------------
figure('Name','3D — Reference clipped at interception');
plot3(X_ref(1,1:sim_end_k), X_ref(2,1:sim_end_k), X_ref(3,1:sim_end_k), ...
      'b','LineWidth',2.0); hold on;
plot3(X_follower_mpc(1,1:plot_end_k), X_follower_mpc(2,1:plot_end_k), ...
      X_follower_mpc(3,1:plot_end_k),'r--','LineWidth',1.5);
plot3(X_follower_actual(1,1:plot_end_k), X_follower_actual(2,1:plot_end_k), ...
      X_follower_actual(3,1:plot_end_k),'m:','LineWidth',1.8);
plot3(ref_pos(1,1), ref_pos(2,1), ref_pos(3,1), ...
      'bs','MarkerSize',10,'MarkerFaceColor','b');
plot3(X_follower_actual(1,1), X_follower_actual(2,1), X_follower_actual(3,1), ...
      'ro','MarkerSize',10,'MarkerFaceColor','r');
plot3(X_follower_actual(1,sim_end_k), X_follower_actual(2,sim_end_k), ...
      X_follower_actual(3,sim_end_k), ...
      'g^','MarkerSize',14,'LineWidth',2,'MarkerFaceColor','g');
grid on; axis equal;
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
title('3D — Reference clipped at confirmed interception');
legend('Reference (clipped)','Follower MPC internal','Follower actual', ...
       'Ref start','Follower start', ...
       sprintf('Confirmed (t=%.2fs)', t_confirmed),'Location','best');
view(45,30);

%% -----------------------------------------------------------------------
%  Plot 3 — Position x/y/z over time
% ------------------------------------------------------------------------
pos_labels = {'x [m]','y [m]','z [m]'};
figure('Name','Position — clipped to interception + buffer');
for i = 1:3
    subplot(3,1,i);
    plot(t_full(1:sim_end_k), X_ref(i,1:sim_end_k), ...
         'b','LineWidth',1.8); hold on;
    plot(t_plot, X_follower_mpc(i,1:plot_end_k),    'r--','LineWidth',1.5);
    plot(t_plot, X_follower_actual(i,1:plot_end_k), 'm:', 'LineWidth',1.8);
    xregion(t_capture_start, t_confirmed, ...
            'FaceColor','g','FaceAlpha',0.15,'EdgeColor','none');
    xline(t_capture_start,'g--', ...
          sprintf('Entered r<%.2fm', R_capture), ...
          'LineWidth',1.0,'LabelVerticalAlignment','bottom');
    xline(t_confirmed,'g-','Confirmed','LineWidth',1.2);
    grid on;
    xlabel('t [s]'); ylabel(pos_labels{i});
    title(pos_labels{i});
    legend('Reference','Follower MPC','Follower actual', ...
           sprintf('Dwell (%.1fs)',T_dwell),'Location','best');
end
sgtitle(sprintf('Position — Interception confirmed at t = %.2f s', t_confirmed));

%% -----------------------------------------------------------------------
%  Plot 4 — Separation distance
% ------------------------------------------------------------------------
dist_vec = vecnorm(X_ref(1:3,1:plot_end_k) - X_follower_actual(1:3,1:plot_end_k), 2, 1);
figure('Name','Separation Distance');
plot(t_plot, dist_vec,'g','LineWidth',1.5); hold on; grid on;
xlabel('t [s]'); ylabel('Distance [m]');
title(sprintf('Separation: Reference vs Follower — Confirmed at t = %.2f s', t_confirmed));
yline(R_capture,'r--',sprintf('Capture radius %.2f m', R_capture),'LineWidth',1.0);
xline(t_capture_start,'g--', ...
      sprintf('Entered radius (t=%.2fs)', t_capture_start),'LineWidth',1.0);
xline(t_confirmed,'g-', ...
      sprintf('Confirmed (t=%.2fs)', t_confirmed),'LineWidth',1.2);
yl = ylim;
fill([t_capture_start t_confirmed t_confirmed t_capture_start], ...
     [yl(1) yl(1) yl(2) yl(2)],'g','FaceAlpha',0.12,'EdgeColor','none');

%% -----------------------------------------------------------------------
%  Plot 5 — Follower control inputs
% ------------------------------------------------------------------------
input_names = {'u1 (thrust)','u2 (roll moment)','u3 (pitch moment)','u4 (yaw moment)'};
figure('Name','Follower Control Inputs');
for i = 1:4
    subplot(2,2,i);
    plot(t_plot, U_follower(i,1:plot_end_k),'r--','LineWidth',1.2); hold on;
    xline(t_confirmed,'g-','Confirmed','LineWidth',1.0);
    grid on;
    xlabel('t [s]'); ylabel(input_names{i});
    title(input_names{i});
    legend('Follower (MPC)','Confirmed','Location','best');
end
sgtitle(sprintf('Follower Control Inputs — Confirmed at t = %.2f s', t_confirmed));

%% -----------------------------------------------------------------------
%  Plot 6 — All 12 follower states: MPC internal vs actual
% ------------------------------------------------------------------------
state_names = {'x','y','z','vx','vy','vz','phi','theta','psi','p','q','r'};
figure('Name','All 12 Follower States — MPC vs Actual');
for i = 1:12
    subplot(4,3,i);
    plot(t_plot, X_follower_mpc(i,1:plot_end_k),    'r--','LineWidth',1.2); hold on;
    plot(t_plot, X_follower_actual(i,1:plot_end_k), 'm:', 'LineWidth',1.5);
    xline(t_confirmed,'g-','LineWidth',1.0);
    grid on;
    xlabel('t [s]'); ylabel(state_names{i});
    title(state_names{i});
end
legend('MPC internal','Follower actual','Confirmed','Location','best');
sgtitle(sprintf('Follower States — Confirmed at t = %.2f s', t_confirmed));

end
