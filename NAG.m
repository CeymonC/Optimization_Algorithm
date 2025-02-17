%% NAG_Method
% (4) Solving using NAG method with fixed step size
function NAG(f, grad_f, x, epsilon, max_iter, x_limits)
    tic
    x1 = x;
    % Initialize momentum variables
    v1 = 0; % Momentum in x1 direction
    % Initialize counters
    f_caltimes = zeros(1, max_iter); % Number of objective function calculations
    grad_caltimes = zeros(1, max_iter); % Number of gradient calculations
    f_val = zeros(1, max_iter); % Objective function value per iteration
    alpha = 0.01; % Fixed step size
    beta = 0.9; % Momentum factor
    caltimes = 0;
        grad_caltimes = 0;
        f_cal_loc = 0;
    % Iterative optimization
    cal_times = 0; % Iteration counter
    for k = 1:max_iter
    % Predict position
        x1_pred = x1 - alpha * beta * v1;
        x_pred = x1_pred;
    % Compute current gradient (evaluated at predicted position)
        grad = grad_f(x_pred);
        grad_caltimes = grad_caltimes + 1; % Increment gradient calculation count
    % Update momentum
        v1 = beta * v1 + (1 - beta) * grad(1);
    % Update
        x1 = x1 - alpha * v1;
        x = x1;
        x = max(min(x, x_limits(:,2)), x_limits(:,1));
        x1 = x(1);
        grad = grad_f(x);
        grad_caltimes = grad_caltimes + 1; % Increment gradient calculation count
    % Compute objective function value
        f_val(k) = f(x);
        caltimes = caltimes + 1;
        f_caltimes(k) =caltimes; % Increment objective function calculation count
        cal_times = k; % Update iteration count
    % Check if gradient is below threshold epsilon; exit if satisfied
        if norm(grad) < epsilon
            T = toc;
            fprintf('%-30s  |     %-6.3f   |     %-6.3f   |     %-6d    |     %-6d    |    %-6.4f    | \n','NAG_Method', x1, f_val(k), f_caltimes(k), grad_caltimes, T);
            break;
        end
    end
% Plot the objective function change
    figure;plot(f_caltimes(1:cal_times), f_val(1:cal_times),"-*");xlabel('Iteration k');ylabel('f(x^{(k)})');title('NAG method with fixed step size');
end