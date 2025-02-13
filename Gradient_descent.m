%% Gradient_descent
% One-dimensional search using fminbnd()
function Gradient_descent(f, df, x, epsilon, maxIter, x_limits)
    tic;
    alpha_max = 1; 
    alpha_min = 0; 
    cal_times = 0;
    f_val = [];
    options = optimset('OutputFcn', @outfun);
    grad = df(x);    % Calculate the gradient at the current point
    grad_caltimes = 1;
    % Gradient descent method
    for iter = 1:maxIter
        grad_caltimes = grad_caltimes + 1;
        % Use fminbnd to perform exact line search, objective function: f(x + alpha * direction)
        options = optimset('OutputFcn', @outfun);
        objFun = @(alpha) f(x - alpha * grad);
        alpha_opt = fminbnd(objFun, alpha_min, alpha_max, options);
        x = x - alpha_opt * grad;    % Update x
        x = max(min(x, x_limits(:,2)), x_limits(:,1));
        cal_times = cal_times + 1;
        grad = df(x);    % Calculate the gradient at the current point
        grad_caltimes = grad_caltimes + 1;
        gradNorm = norm(grad);    % Norm of the gradient
        % Check if termination condition is met (gradNorm < epsilon)
        if gradNorm <= epsilon
            break;
        end
    end
    T = toc;
    fprintf('%-30s  |     %-6.3f   |     %-6.3f   |     %-6d    |     %-6d    |    %-6.3f    | \n','Gradient_descent', x, f_val(end), length(f_val), grad_caltimes, T);
    figure;plot(1:length(f_val),f_val,"-*");xlabel("k");ylabel("f(x^{(k)})");title("Gradient Method with Exact Line Search");
    function stop = outfun(~, optimValues, state)
        stop = false;
        if isequal(state, 'iter')
            f_val = [f_val; optimValues.fval];
        end
    end
end