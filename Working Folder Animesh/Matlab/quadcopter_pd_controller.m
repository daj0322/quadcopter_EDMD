function u = quadcopter_pd_controller(x, x_ref, Kp, Kd, m, g)
% QUADCOPTER_PD_CONTROLLER  PD controller for linearized quadcopter hover.
%
% u = quadcopter_pd_controller(x, x_ref, Kp, Kd, m, g)
%
% INPUTS:
%   x      - 12x1 current state  [pW; vW; eta; omegaB]
%   x_ref  - 12x1 reference state
%   Kp     - 4x12 proportional gain matrix  (or scalar -> Kp*eye(4,12))
%   Kd     - 4x12 derivative gain matrix    (or scalar -> Kd*eye(4,12))
%   m      - mass [kg]
%   g      - gravitational acceleration [m/s^2]
%
% OUTPUT:
%   u - 4x1 total control input
%
% CONTROL LAW:
%   u = u_star + Kp*e_p + Kd*e_d
%
% NOTE:
%   delta_u = u - u_star is computed inside quadcopter_dynamics,
%   consistent with:  x_dot = A*x + B*delta_u

%% --- Input validation ---
arguments
    x      (12,1) double
    x_ref  (12,1) double
    Kp
    Kd
    m      (1,1) double {mustBePositive}
    g      (1,1) double {mustBePositive}
end

%% --- Expand scalar gains ---
if isscalar(Kp), Kp = Kp * eye(4,12); end
if isscalar(Kd), Kd = Kd * eye(4,12); end

assert(isequal(size(Kp), [4,12]), 'Kp must be 4x12 or scalar');
assert(isequal(size(Kd), [4,12]), 'Kd must be 4x12 or scalar');

%% --- Hover equilibrium ---
u_star = [m*g; 0; 0; 0];

%% --- State error ---
e = x_ref - x;

%% --- Proportional and derivative error components ---
e_p = [e(1:3);    zeros(3,1); e(7:9);    zeros(3,1)];  % position + attitude
e_d = [zeros(3,1); e(4:6);   zeros(3,1); e(10:12)  ];  % velocity + angular rate

%% --- Total control input ---
u = u_star + Kp * e_p + Kd * e_d;

end