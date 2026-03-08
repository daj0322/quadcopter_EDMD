function x_dot = quadcopter_dynamics(x, u, A, B, m, g)
% QUADCOPTER_DYNAMICS  Linearized quadcopter dynamics.
%
% x_dot = quadcopter_dynamics(x, u, A, B, m, g)
%
% INPUTS:
%   x  - 12x1 current state
%   u  - 4x1  total control input (from controller)
%   A  - 12x12 state matrix
%   B  - 12x4  input matrix
%   m  - mass [kg]
%   g  - gravitational acceleration [m/s^2]
%
% OUTPUT:
%   x_dot - 12x1 state derivative
%
% delta_u = u - u_star is computed here, consistent with the paper:
%   x_dot = A*x + B*delta_u

%% --- Hover equilibrium ---
u_star = [m*g; 0; 0; 0];

%% --- Control perturbation (paper: delta_u = u - u_star) ---
delta_u = u - u_star;

%% --- Linearized dynamics (Eq. 29) ---
x_dot = A*x + B*delta_u;

end