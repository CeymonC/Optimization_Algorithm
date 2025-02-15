%% bfgs_optimization
% (3) BFGS method based on fixed step size
function [x_opt, f_opt] = bfgs_optimization(f, grad_f, x0, epsilon, maxIter, x_limits)
    tic
    f_caltimes = 0;
    f_cal_loc = 0;
    grad_caltimes = 0;
    caltimes = 0;
    % Initial approximation of Hessian matrix (BFGS initial value can be the identity matrix)
    H = eye(length(x0));  % Hessian matrix approximation
    % Initial gradient
    grad = grad_f(x0);
    grad_caltimes = grad_caltimes + 1;
    % Current solution
    x = x0;
    cal_times = 0;
    % BFGS algorithm iteration
    for iter = 1:maxIter
        alpha_opt = 0.01;
        p = -H * grad;        % Calculate search direction: p = -H * grad
        x_new = x + alpha_opt * p;        % Update solution
% Calculate new gradient
        grad_new = grad_f(x_new);
        grad_caltimes = grad_caltimes + 1;

% Compute gradient difference
        s = x_new - x;  % Difference between current and previous positions
        y = grad_new - grad;    % Difference between current and previous gradients
% Compute BFGS correction (updated Hessian matrix approximation formula)
        I = eye(2);          % Identity matrix
        H = H + (s * s') / (y' * s) - (H * y * y' * H) / (y' * H * y);
% Update solution and gradient
        x = x_new;
        x = max(min(x, x_limits(:,2)), x_limits(:,1));
        grad = grad_f(x);
        grad_caltimes = grad_caltimes + 1;
        f_cal_loc = f_cal_loc + 1;
        caltimes = caltimes + 1;
        f_val(f_cal_loc) = f(x);   
        f_caltimes(f_cal_loc) = iter;
        % Check for convergence
        if norm(grad) <= epsilon
            break;end
    end
    T = toc;
    % Optimal solution and value
    x_opt = x;
    f_opt = f(x_opt);
    % Output results
    fprintf('%-30s  |     %-6.3f   |     %-6.3f   |     %-6d    |     %-6d    |    %-6.4f    | \n','BFGS_Optimization', x_opt, f_opt, f_caltimes(caltimes), grad_caltimes, T);
    figure;plot(f_caltimes(1:caltimes),f_val(1:f_cal_loc),"-*");xlabel("k");ylabel("f(x^{(k)})");title("BFGS Method with Fixed Step Size");
end