%% Adam_Method
% (5) Adam method solution based on fixed step size
function Adam(f, grad_f, x, epsilon, max_iter, x_limits)
    tic;
    % Initialize parameters for the Adam algorithm
    m = zeros(1, 1);      % First moment estimate
    v = zeros(1, 1);      % Second moment estimate
    t = 0;                % Time step
    eps = 1e-8;           % Small constant to prevent division by zero
    beta1 = 0.9;          % Decay rate for first moment
    beta2 = 0.999;        % Decay rate for second moment
    alpha = 0.01;         % Fixed step size
    caltimes = 0;
    grad_caltimes = 0;
    g = grad_f(x);  % Compute gradient
    grad_caltimes = grad_caltimes + 1;
    % Start iterative optimization
    for iter = 1:max_iter
        t = t + 1;  % Update time step
    % Adam parameter updates
        m = beta1 * m + (1 - beta1) * g;  % Update first moment
        v = beta2 * v + (1 - beta2) * g^2;  % Update second moment
    % Bias correction
        m_hat = m / (1 - beta1^t);  % Correct first moment
        v_hat = v / (1 - beta2^t);  % Correct second moment
    % Update parameters using corrected moments
        x = x - alpha * m_hat ./ (sqrt(v_hat) + eps);  % Use m_hat and v_hat
        x = max(min(x, x_limits(:,2)), x_limits(:,1));
        g = grad_f(x);
        grad_caltimes = grad_caltimes + 1;
        caltimes = caltimes + 1;
        f_val(caltimes) = f(x);
        f_caltimes(caltimes) = caltimes;
    % Calculate gradient norm and check termination condition
        grad_norm = norm(g);
        if grad_norm <= epsilon
            break;
        end
    end
    T = toc;
% Output results
    fprintf('%-30s  |     %-6.3f   |     %-6.3f   |     %-6d    |     %-6d    |    %-6.4f    | \n','Adam_Method', x, f_val(caltimes), f_caltimes(caltimes), grad_caltimes, T);
% Plot objective function changes
    figure;plot(f_caltimes(1:caltimes), f_val(1:caltimes),"-*");xlabel('Iteration count k');ylabel('f(x^{(k)})');title('Adam Method with Fixed Step Size');
end