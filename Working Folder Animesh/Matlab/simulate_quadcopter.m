function simulate_quadcopter()

%% --- Parameters ---
m = 1.0; g = 9.81;
Ixx = 0.01; Iyy = 0.01; Izz = 0.02;
kv = 0.1; kw = 0.01;
Ts = 0.01;

%% --- Build model ---
[A, B, ~, ~] = quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts);

%% --- Gains ---
Kp = zeros(4,12);
Kd = zeros(4,12);

% z position -> thrust
Kp(1,3)  =  2.0;  Kd(1,6)  = 1.5;

% y error -> roll (phi) -> vy  (note: y_dot ~ -g*phi, so negative sign)
Kp(2,2)  = -0.5;  Kd(2,5)  = -0.3;   % y pos + vy -> u2 (roll moment)
Kp(2,7)  =  3.0;  Kd(2,10) =  0.5;   % phi + p    -> u2

% x error -> pitch (theta) -> vx  (x_dot ~ g*theta, positive sign)
Kp(3,1)  =  0.5;  Kd(3,4)  =  0.3;   % x pos + vx -> u3 (pitch moment)
Kp(3,8)  =  3.0;  Kd(3,11) =  0.5;   % theta + q  -> u3

% psi -> yaw
Kp(4,9)  =  1.0;  Kd(4,12) =  0.3;

%The coupling comes directly from the G matrix in the paper:
%v_x_dot =  g*theta;   %→  to move in x, pitch forward  (u3)
%v_y_dot = -g*phi;     %→  to move in y, roll sideways  (u2, negative sign)
%% --- Initial and reference states ---
x     = zeros(12,1);
x_ref = zeros(12,1);
x_ref(1:3) = [1; 1; 1];   % target position x=1, y=1, z=1

%% --- Simulation loop (Euler integration) ---
t_end = 10;
t     = 0:Ts:t_end;
X     = zeros(12, length(t));
U     = zeros(4,  length(t));

for k = 1:length(t)
    X(:,k) = x;

    % Controller output feeds directly into dynamics
    u      = quadcopter_pd_controller(x, x_ref, Kp, Kd, m, g);
    U(:,k) = u;
    x_dot  = quadcopter_dynamics(x, u, A, B, m, g);

    % Euler step
    x = x + Ts * x_dot;
end

%% --- Plot states ---
state_names = {'x','y','z','vx','vy','vz','phi','theta','psi','p','q','r'};
ref_vals    = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];   % full reference state

figure('Name','Quadcopter States');
for i = 1:12
    subplot(4,3,i);
    plot(t, X(i,:), 'b', 'LineWidth', 1.2); hold on;
    yline(ref_vals(i), 'r--', 'LineWidth', 1.0);
    grid on;
    xlabel('t [s]'); ylabel(state_names{i});
    title(state_names{i});
end
sgtitle('Quadcopter PD Controller — States (ref = [1,1,1])');
legend('state','reference','Location','best');

%% --- Plot control inputs ---
input_names = {'u1 (thrust)','u2 (roll)','u3 (pitch)','u4 (yaw)'};
figure('Name','Control Inputs');
for i = 1:4
    subplot(2,2,i);
    plot(t, U(i,:), 'k', 'LineWidth', 1.2); grid on;
    xlabel('t [s]'); ylabel(input_names{i});
    title(input_names{i});
end
sgtitle('Quadcopter PD Controller — Control Inputs');

end