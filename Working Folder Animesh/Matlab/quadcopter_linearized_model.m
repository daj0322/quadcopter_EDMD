function [A, B, sys_c, sys_d] = quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts)

%QUADCOPTER_LINEARIZED_MODEL  Continuous & discrete linearized hover model.
%
% [A, B, sys_c, sys_d] = quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts)
%
% INPUTS:
%   m    - mass [kg]
%   g    - gravitational acceleration [m/s^2]
%   Ixx  - moment of inertia about x-axis [kg·m^2]
%   Iyy  - moment of inertia about y-axis [kg·m^2]
%   Izz  - moment of inertia about z-axis [kg·m^2]
%   kv   - translational drag coefficient [N·s/m]
%   kw   - angular drag coefficient [N·m·s/rad]
%   Ts   - sample time for discretization [s]  (use [] to skip)
%
% OUTPUTS:
%   A      - 12x12 continuous-time state matrix
%   B      - 12x4  continuous-time input matrix
%   sys_c  - continuous-time ss object
%   sys_d  - discrete-time ss object (ZOH), empty if Ts is []
%
% STATE:  x = [pW(3); vW(3); eta(3); omegaB(3)]
%             = [x y z | vx vy vz | phi theta psi | p q r]
%
% INPUT:  delta_u = u - u_star,  u_star = [mg, 0, 0, 0]^T
%         delta_u is computed in quadcopter_dynamics, not here.
%
% EXAMPLE:
%   [A, B, sys_c, sys_d] = quadcopter_linearized_model(1.0, 9.81, ...
%                           0.01, 0.01, 0.02, 0.1, 0.01, 0.01);

% --- Input validation ---
arguments
    m    (1,1) double {mustBePositive}
    g    (1,1) double {mustBePositive}
    Ixx  (1,1) double {mustBePositive}
    Iyy  (1,1) double {mustBePositive}
    Izz  (1,1) double {mustBePositive}
    kv   (1,1) double {mustBeNonnegative}
    kw   (1,1) double {mustBeNonnegative}
    Ts            = []
end

% --- Damping rates (Eq. 28) ---
dv = kv / m;
Dw = diag([kw/Ixx, kw/Iyy, kw/Izz]);

% --- Gravity-attitude coupling (Eq. 32) ---
G = [0, g, 0;
    -g, 0, 0;
     0, 0, 0];

% --- Input sub-matrices (Eq. 31) ---
Bv = [0,   0,     0,     0;
      0,   0,     0,     0;
      1/m, 0,     0,     0];

Bw = [0,   1/Ixx, 0,     0;
      0,   0,     1/Iyy, 0;
      0,   0,     0,     1/Izz];

% --- State matrix A (Eq. 30) ---
O3 = zeros(3,3);
I3 = eye(3);

A = [O3,    I3,      O3,  O3;
     O3, -dv*I3,      G,  O3;
     O3,    O3,      O3,  I3;
     O3,    O3,      O3, -Dw];

% --- Input matrix B (Eq. 31) ---
O34 = zeros(3,4);

B = [O34;
     Bv;
     O34;
     Bw];

% --- State-space object (paper only defines x_dot = Ax + B*delta_u) ---
state_names = {'x','y','z','vx','vy','vz','phi','theta','psi','p','q','r'};
input_names = {'delta_u1','delta_u2','delta_u3','delta_u4'};

sys_c = ss(A, B, [], []);
sys_c.StateName = state_names;
sys_c.InputName = input_names;

% --- Discretization (ZOH) ---
if ~isempty(Ts)
    validateattributes(Ts, {'double'}, {'scalar','positive'}, ...
                       'quadcopter_linearized_model', 'Ts');
    sys_d = c2d(sys_c, Ts, 'zoh');
else
    sys_d = [];
end

end