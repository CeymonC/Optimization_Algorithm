%% GA
clear;
best_f = -418.982887;
% Range of independent variables
range_x=[ones(1,1),-ones(1,1)]*500;
% Dimension
n=size(range_x,1);
% Population size
gn=400;
% Number of individuals entering the next generation
m=50;
% Number of iterations
times=200;
% Randomly generate initial population
group=zeros(n,gn);
for k=1:n
    group(k,:)=(rand(1,gn))*(range_x(k,2)-range_x(k,1))+range_x(k,1);
end
% Initialize best solution record
best_value=zeros(1,times);
tic;
for k=1:times
    y=f(group);
    % Convert all to positive values
    if min(y)<0
        tem=y-min(y)*1.0001;
    else
        tem=y+0.1;
    end
    % Smaller values indicate better fitness
    tem=1./tem;
    child=zeros(n,gn);
    % Select m individuals for next generation
    for i=1:m
        % Roulette wheel selection: higher fitness has higher probability
        temp=zeros(1,gn-i+1);
        for j=1:gn-i+1
            temp(j)=sum(tem(1:j));
        end
        temp=temp/temp(gn-i+1);
        % Keep the fittest individual
        choose=find(temp>rand(1),1);
        child(:,i)=group(:,choose);
        group=[group(:,1:choose-1),group(:,choose+1:end)];
        tem=[tem(1:choose-1),tem(choose+1:end)];
    end
    % Chromosomal crossover: gene recombination occurs when retained individuals produce offspring
    for i=1:floor((gn-m)/2)
        exchange=randperm(m,2);
        a=rand(n,1);
        child(:,i*2-1+m)=a.*child(:,exchange(1))+(1-a).*child(:,exchange(2));
        child(:,i*2+m)=(1-a).*child(:,exchange(1))+a.*child(:,exchange(2));
    end
    if mod(gn-m,2)==1
        exchange=randperm(m,2);
        child(:,gn)=(child(:,exchange(1))+child(:,exchange(2)))/2;
    end
    % Chromosomal mutation may occur during gene recombination
    if rand(1)<0.1
        exchange=randperm(gn-m,1);
        a=rand(1);
        for j=1:n
            child(j,exchange+m)=a.*child(j,exchange+m)+(1-a).*(rand(1)*(range_x(j,2)-range_x(j,1))+range_x(j,1));
        end
    end
    % Update population with new offspring
    group=child;
    best_value(k)=min(f(group));
    if k>5&&abs(best_value(k)-best_value(k-5))<1e-5
        break;
    end
end
time=toc;
% Display execution time
[mini,index]=min(f(group));
% Display optimization results
for k=1:n
    % Display variable values
end
fprintf('%-30s  |     %-6.3f   |     %-6.3f   |    %-6.4f    |    %-6.4f    | \n','GA', group(k,index), mini, time, (mini-best_f)/best_f);