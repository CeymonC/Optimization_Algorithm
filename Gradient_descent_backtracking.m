% gradient_descent_backtracking
% (2) Gradient descent method using backtracking line search (inexact 1D search)
function [x_opt, f_opt] = gradient_descent_backtracking(f, grad_f, x, epsilon, max_iter, x_limits)
    tic;
    % Backtracking parameters
    alpha0 = 1;       % Initial step size
    omega = 0.7;      % Step size reduction factor (0 < omega < 1)
    rho = 0.4;        % Constant in Armijo condition (typically small)
    calc_count = 0;
    grad_calc_count = 0;
    f_eval_count = 0;
    
    grad = grad_f(x);              % Compute gradient at current point
    grad_calc_count = grad_calc_count + 1;
    
    % Iterative solving
    for iter = 1:max_iter
        alpha = alpha0;            % Reset step size
        
        % Check Armijo condition
        f_current = f(x);
        f_trial = f(x - alpha * grad);
        while f_trial > f_current - rho * alpha * grad' * grad
            alpha = omega * alpha; % Reduce step size
            f_current = f(x);
            f_trial = f(x - alpha * grad);
        end
        
        % Update position
        x = x - alpha * grad;
        x = max(min(x, x_limits(:,2)), x_limits(:,1)); % Apply box constraints
        grad = grad_f(x);          % Compute new gradient
        grad_calc_count = grad_calc_count + 1;
        
        grad_norm = norm(grad);    % Compute gradient norm
        
        % Check convergence condition
        if grad_norm <= epsilon
            break;
        end
        
        f_eval_count = f_eval_count + 1;
        calc_count = calc_count + 1;
        f_history(f_eval_count) = f_current;
        eval_counts(f_eval_count) = calc_count;
    end
    
    exec_time = toc;
    
    % Output iteration information
    fprintf('%-30s  |     %-6.3f   |     %-6.3f   |     %-6d    |     %-6d    |    %-6.4f    | \n',...
            'grad_desc_backtrack', x, f_history(f_eval_count), eval_counts(f_eval_count), grad_calc_count, exec_time);
    
    % Plot objective function progression
    figure;
    plot(eval_counts(1:f_eval_count), f_history(1:f_eval_count), "-*");
    xlabel('Iteration count k');
    ylabel('f(x^{(k)})');
    title('Gradient Method with Backtracking Line Search');
end