clear
close all

%% 1. Load the data
sample_time = 0.05;
length = 100;
simulation_time = sample_time * length;

n1 = 50;
dimension = 2;

for i = 1 : n1
    initial(i).init_dq = [0.8/i; -0.5/i];
    initial(i).init_q = [0; 0];
end

for i = 1 : n1
    init_dq = initial(i).init_dq;
    init_q = initial(i).init_q;
    
    data_from_simulink = sim('circular_data');

    de_values = data_from_simulink.de.signals.values;
    dde_values = data_from_simulink.dde.signals.values;
    
    derivative_training_sample(i).data = squeeze(de_values)';
    derivative_derivative_training_sample(i).data = squeeze(dde_values)';
end

%% 2. Define Lyapunov Candidate V(de)（Complete 8th Order Polynomial）
syms de1 de2
de = [de1; de2];

% Define 42 symbolic coefficients
a = sym('a', [42,1]);

% Construct the complete 8th-order polynomial for V(de)
V_sym = ...
    a(1) * de1^2 + a(2) * de2^2 + a(3) * de1 * de2 + ...
    a(4) * de1^3 + a(5) * de2^3 + a(6) * de1^2 * de2 + a(7) * de1 * de2^2 + ...
    a(8) * de1^4 + a(9) * de2^4 + a(10) * de1^3 * de2 + a(11) * de1 * de2^3 + a(12) * de1^2 * de2^2 + ...
    a(13) * de1^5 + a(14) * de2^5 + a(15) * de1^4 * de2 + a(16) * de1 * de2^4 + a(17) * de1^3 * de2^2 + a(18) * de1^2 * de2^3 + ...
    a(19) * de1^6 + a(20) * de2^6 + a(21) * de1^5 * de2 + a(22) * de1 * de2^5 + a(23) * de1^4 * de2^2 + a(24) * de1^2 * de2^4 + a(25) * de1^3 * de2^3 + ...
    a(26) * de1^7 + a(27) * de2^7 + a(28) * de1^6 * de2 + a(29) * de1 * de2^6 + a(30) * de1^5 * de2^2 + a(31) * de1^2 * de2^5 + a(32) * de1^4 * de2^3 + a(33) * de1^3 * de2^4 + ...
    a(34) * de1^8 + a(35) * de2^8 + a(36) * de1^7 * de2 + a(37) * de1 * de2^7 + a(38) * de1^6 * de2^2 + a(39) * de1^2 * de2^6 + a(40) * de1^5 * de2^3 + a(41) * de1^3 * de2^5 + a(42) * de1^4 * de2^4;

% Compute symbolic gradient dV = [∂V/∂de1, ∂V/∂de2]
dV_sym = jacobian(V_sym, de);

%% 3. Define λ and γ
lambda = sym('lambda');
gamma = 0.0;  % Fixed constant

% Define optimization variables
x = [a; lambda];

% Convert symbolic V and dV to MATLAB function handles
V_fun = matlabFunction(V_sym, 'Vars', {de, a});
dV_fun = matlabFunction(dV_sym, 'Vars', {de, a});

% Objective function (no objective, just optimization constraints)
obj_fun = @(x) 0;

%% 4. Define bounds
x0 = [ 1.6555
    2.8821
    0.6178
    0.8305
    0.2766
   -0.8001
   -0.4364
    0.0938
    0.9189
    0.9298
   -0.6842
    0.9413
    0.9143
   -0.0220
    0.6006
   -0.7153
   -0.1565
    0.8316
    0.5844
    0.9322
    0.3115
   -0.9268
    0.6983
    0.8682
    0.3575
    0.5155
    0.5094
   -0.2155
    0.3140
   -0.6576
    0.4125
   -0.9363
   -0.4461
   -0.9077
   -0.7659
    0.6469
    0.3950
   -0.3658
    0.9012
   -0.9311
   -0.1224
   -0.2369
    1.7691];
lb = [-10 * ones(42, 1); 0.001];  % λ should be positive
ub = [10 * ones(42, 1); 10];   % Upper bound

%% 5. Run fmincon Optimization
options = optimoptions('fmincon', 'Algorithm', 'sqp', ...
    'Display', 'iter', 'StepTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-6, 'MaxIterations', 1000);
sol = fmincon(obj_fun, x0, [], [], [], [], lb, ub, @(x) nonlinear_constraints(x, n1, length, derivative_training_sample, derivative_derivative_training_sample, gamma, V_fun, dV_fun), options);

%% 6. Output the result
if ~isempty(sol)
    fprintf('✅ Optimization Success！\n');
    fprintf('V(de) = ');
    for i = 1:42
        if i < 42
            fprintf('%.4f * term%d + ', sol(i), i);
        else
            fprintf('%.4f * term%d\n', sol(i), i);
        end
    end
    fprintf('\nλ_opt = %.4f\n', sol(43));
    fprintf('γ (fixed) = %.4f\n', gamma);
else
    disp('❌ Optimization Failed');
end

%% 7. Define Constraints
function [c, ceq] = nonlinear_constraints(x, n1, length, derivative_training_sample, derivative_derivative_training_sample, gamma, V_fun, dV_fun)
    a = x(1:42);
    lambda = x(43);
    
    c = [];
    for i = 1 : n1
        for t = 1 : length
            de_val = derivative_training_sample(i).data(t, :)';    % de = [de1; de2]
            dde_val = derivative_derivative_training_sample(i).data(t, :)';  % dde = [dde1; dde2]

            % Compute dV = ∇V * dde
            dV_val = dV_fun(de_val, a);
            dV = dV_val(:)' * dde_val;

            % Constraints: dV + λV + γ ≤ 0 and V > 0
            c = [c; dV + lambda * V_fun(de_val, a) + gamma; -V_fun(de_val, a)];
        end
    end
    ceq = [];
end

%%
[X1,X2] = meshgrid(-1:0.1:1,-1:0.1:1);

for i=size(X1,1):-1:1
    for j = size(X1,2):-1:1
    V_val(i,j) = V_fun([X1(i,j); X2(i,j)], sol(1:end-1));
    %dV_val(i,j) = dV_fun([X1(i,j); X2(i,j)], sol(1:end-1));
    end
end
figure
surf(X1,X2,V_val)