%%ACO
clear;
best_f = -418.982887;
% Range of independent variables
range_x = [ones(1,1), -ones(1,1)] * 500;
% Dimension
n = size(range_x, 1);
% Number of ants
m = 1000;
% Number of iterations
times = 100;
% Pheromone evaporation coefficient
rho = 0.8;
% Transition probability constant
p0 = 1;
% Transition probability
p = zeros(1, m);
% x is the initial position of the ant colony
x = zeros(n, m);
for k = 1:n
x(k, :) = (rand(1, m)) * (range_x(k, 2) - range_x(k, 1)) + range_x(k, 1);
end
% tau represents pheromone
tau = -f(x);
% Set current optimal solution
best_value = zeros(1, times);
tic;
for i = 1:times
[~, bestindex] = max(tau);
for j = 1:m
% The higher the pheromone, the less likely to transfer
p(j) = (tau(bestindex) - tau(j)) / tau(bestindex);
if p(j) < p0
% Smaller transfer step for higher pheromone
temp = x(:, j) + (rand(n, 1) * 2 - 1) / i;
else
temp = zeros(n, 1);
% Larger transfer step for lower pheromone
for k = 1:n
temp(k) = x(k, j) + (rand(1, 1) - 0.5) * (range(k, 2) - range(k, 1));
end
end
% Apply boundary constraints
for k = 1:n
if temp(k) < range_x(k, 1)
temp(k) = range_x(k, 1);
end
if temp(k) > range_x(k, 2)
temp(k) = range_x(k, 2);
end
end
% Move if the value decreases after transfer
if f(temp) < f(x(:, j))
x(:, j) = temp;
end
% Update pheromone (smaller function value → higher pheromone)
tau(j) = (1 - rho) * tau(j) - f(x(:, j));
end
best_value(i) = min(f(x));
if i > 5 && abs(best_value(i) - best_value(i - 5)) < 1e-5
break;
end
end
time = toc;
% disp(['Time used: ', num2str(time), ' seconds'])
[mini, index] = min(f(x));
% disp(['fmin=', num2str(mini)]);
for k = 1:n
% disp(['x', num2str(k), '=', num2str(x(k, index))]);
end
fprintf('%-30s | %-6.3f | %-6.3f | %-6.4f | %-6.4f | \n', 'ACO', x(k, index), mini, time, (mini - best_f)/best_f);