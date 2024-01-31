% Initialize vectors to hold the timings
a_n = []; 
b_n = []; 
ns = 200:50:1000; 

for n = ns
    x = 2 * ones(n, 1); % Vector of all twos
    b = ones(n, 1); % Vector of all ones
    H_sparse = rosenbrockHessian(x); 

    tic;
    x_sparse = H_sparse \ b;
    a_n(end+1) = toc;

    H_dense = full(H_sparse); 
    tic;
    x_dense = H_dense \ b;
    b_n(end+1) = toc;

    if b_n(end) > 60
        fprintf('Dense solve exceeded 1 minute at n = %d\n', n);
        break;
    end
end

% Plot the results
% figure;
% loglog(ns, a_n, 'o-', 'DisplayName', 'Sparse Solve Time');
% hold on;
% loglog(ns, b_n, 'x-', 'DisplayName', 'Dense Solve Time');
% xlabel('Dimension n');
% ylabel('Time (seconds)');
% title('Solve Time for Sparse and Dense Rosenbrock Hessian');
% legend('Location', 'NorthWest');
% grid on;



%-----4 call function -----------
ns = 10:10:200;  

num_systems_solved_NM = zeros(length(ns), 1);
total_time_NM = zeros(length(ns), 1);
num_systems_solved_dogleg = zeros(length(ns), 1);
total_time_dogleg = zeros(length(ns), 1);

gradient_tolerance = 1e-3;
max_iter = 1000;
line_search_params.tol = gradient_tolerance;
hess_mod_params.beta = 1e-3;  
hess_mod_params.epsilon = 1e-8;  
line_search_params.tol = 1e-6; 

% Define parameters for dogleg
delta_max = 2.0;
eta = 0.15;


for i = 1:length(ns)
    n = ns(i);
    x0 = 2 * ones(n, 1); % Initial point of all 2's
    
    % Run NMHM
    tic;
    [x_NM, ~, num_systems_solved] = newton_modified(@rosenbrockFull, x0, max_iter, line_search_params, hess_mod_params);
    total_time_NM(i) = toc;
    num_systems_solved_NM(i) = num_systems_solved;
    
    % Run dogleg trust-region method
    tic;
    [x_dogleg, num_linear_solves, num_func_evals] = dogleg_trust_region(x0, max_iter, delta_max, eta, @rosenbrock);
    total_time_dogleg(i) = toc;
    num_systems_solved_dogleg(i) = num_linear_solves;
end

% Plot the number of linear systems solved
figure;
plot(ns, num_systems_solved_NM, 'b-o', ns, num_systems_solved_dogleg, 'r-x');
xlabel('Problem Size n');
ylabel('Number of Linear Systems Solved');
legend('NMHM', 'Dogleg');
title('Comparison of NMHM and Dogleg Method - Linear Systems Solved');

% Plot the total computation time for linear systems
figure;
plot(ns, total_time_NM, 'b-o', ns, total_time_dogleg, 'r-x');
xlabel('Problem Size n');
ylabel('Total Computation Time (seconds)');
legend('NMHM', 'Dogleg');
title('Comparison of NMHM and Dogleg Method - Computation Time');

% Analysis and discussion of the findings will be based on the plots


%-----4 call function ------



function H = rosenbrockHessian(x)
    n = length(x);
    H = spalloc(n, n, 3*n - 2); 

    for i = 1:n-1
        % Diagonal element (second derivative w.r.t xi)
        H(i, i) = H(i, i) + 1200*x(i)^2 - 400*x(i+1) + 2;

        % Off-diagonal element (second derivative w.r.t xi and xi+1)
        H(i, i+1) = H(i, i+1) - 400*x(i);

        % Since Hessian is symmetric, copy the off-diagonal element
        H(i+1, i) = H(i, i+1);
    end

    H(n, n) = H(n, n) + 200;
end

function [f, g] = rosenbrock(x)
    n = length(x);
    f = 0;
    g = zeros(n, 1);
    for i = 1:(n - 1)
        f = f + 100*(x(i+1) - x(i)^2)^2 + (1 - x(i))^2;
        g(i) = g(i) - 400*x(i)*(x(i+1) - x(i)^2) - 2*(1 - x(i));
        if i < n
            g(i+1) = g(i+1) + 200*(x(i+1) - x(i)^2);
        end
    end
end

function [fval, grad, H] = rosenbrockFull(x)
    [fval, grad] = rosenbrock(x); % Get the function value and gradient
    H = rosenbrockHessian(x);     % Get the Hessian
end


function [x, num_linear_solves, num_func_evals] = dogleg_trust_region(x0, max_iters, delta_max, eta, rosenbrock_func)
    num_linear_solves = 0;
    num_func_evals = 0;
    x = x0;
    delta = delta_max / 2; 
    
    for k = 1:max_iters
        [f, g] = rosenbrock_func(x);
        H = rosenbrockHessian(x);
        num_func_evals = num_func_evals + 1;
        
        % Compute pu
        pu = -((g' * g) / (g' * H * g)) * g;
        
        if norm(pu) >= delta
            p = (delta / norm(pu)) * pu;
        else
            pb = -H\g;
            num_linear_solves = num_linear_solves + 1;

            if norm(pb) <= delta
                p = pb;
            else  
                tau = 2;
                while true
                    p = pu + (tau - 1) * (pb - pu);
                    if norm(p) <= delta
                        break;
                    end
                    tau = tau - 0.1; 
                end
            end
        end
        
        rho = (f - rosenbrock_func(x + p)) / (-g' * p - 0.5 * p' * H * p);
        num_func_evals = num_func_evals + 1;
        
        if rho < 0.25
            delta = 0.25 * delta;
        else
            if rho > 0.75 && norm(p) == delta
                delta = min(2 * delta, delta_max);
            end
        end
        
        if rho > eta
            x = x + p;
        end
        
        % Check for convergence 
        if norm(g) < 1e-6
            break;
        end
    end
end

%----------------3 and 4----------------



function [x, fval, exitflag] = newton_modified(f, x0, max_iter, line_search_params, hess_mod_params)
    x = x0;
    [fval, grad, H] = f(x); 
    exitflag = 1;
    for k = 1:max_iter
        [L, flag] = modified_cholesky(H, hess_mod_params);
        if flag == 0
            exitflag = -1; 
            return;
        end
        
        p = -L' \ (L \ grad);
        alpha = 1;  
        x = x + alpha * p;
        [fval, grad, H] = f(x);

        if norm(grad) < line_search_params.tol
            exitflag = 0; 
            return;
        end
    end
    
end
    
   




function [L, flag] = modified_cholesky(H, params)
    t = 0;
    if min(diag(H)) > 0
        t = 0;
    else
        t = -min(diag(H)) + params.beta;
    end
    
    n = size(H, 1);
    L = zeros(n);
    flag = 0; 
    
    for k = 1:n
        for j = 1:k-1
            H(k, k) = H(k, k) - L(k, j)^2;
        end
        if H(k, k) + t <= params.epsilon
            t = 2 * t + params.beta;
            k = 0;
            continue;
        end
        L(k, k) = sqrt(H(k, k) + t);
        for i = k+1:n
            for j = 1:k-1
                H(i, k) = H(i, k) - L(i, j) * L(k, j);
            end
            L(i, k) = H(i, k) / L(k, k);
        end
    end
    
    if t == 0
        flag = 1; 
    end
end
