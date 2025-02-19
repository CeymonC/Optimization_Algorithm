%% SA
clear;
best_f = -418.982887;
% Variable range
range_x=[ones(1,1),-ones(1,1)]500;
% Dimension
n=size(range_x,1);
% Number of trials
num=10;
value=zeros(n,num);
delta_t=0.2;
tic;
for i=1:num
% Assign initial values to x
x=zeros(n,1);
for k=1:n
x(k)=(rand(1))(range_x(k,2)-range_x(k,1))+range_x(k,1);
end
% Initial temperature t
t=100;
while t>1e-5
x=SA_metripolis(range_x,t,x,n);
% Temperature decreases by delta_t each time
t=t-delta_t;
end
value(:,i)=x;
end
time=toc;
% disp(['Time elapsed: ',num2str(time),' seconds'])
[mini,index]=min(f(value));
% disp(['fmin=',num2str(mini)]);
for k=1:n
% disp(['x',num2str(k),'=',num2str(value(k,index))]);
end
fprintf('%-30s | %-6.3f | %-6.3f | %-6.4f | %-6.4f | \n','SA', value(k,index), mini, time, (mini-best_f)/best_f);