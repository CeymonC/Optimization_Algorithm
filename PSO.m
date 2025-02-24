%% PSO
clear;
best_f = -418.982887;
% Range of independent variables
range_x=[ones(1,1),-ones(1,1)]500;
% Dimension
n=size(range_x,1);
% Number of iterations
times=100;
% w is the inertia weight
w=0.8;
% c1 is the cognitive weight
c1=1;
% c2 is the social weight
c2=1;
% Number of particles in the swarm
gn=1000;
% Initial positions of the particles
x=zeros(n,gn);
for k=1:n
x(k,:)=(rand(1,gn))(range_x(k,2)-range_x(k,1))+range_x(k,1);
end
% Individual best positions
p=x;
% v represents the velocity of the particles
v=zeros(n,gn);
% Set current best solution
best_value=zeros(1,times);
tic;
for i=1:times
[solve,gbest]=min(f(x));
for j=1:gn
% Velocity consists of 3 parts: inertial velocity, individual best and global best
v(:,j)=wv(:,j)+c1rand(n,1).(p(:,j)-x(:,j))+c2rand(n,1).*(x(:,gbest)-x(:,j));
x(:,j)=x(:,j)+v(:,j);
% Limit the solution range
for k=1:n
if x(k,j)<range_x(k,1)
x(k,j)=range_x(k,1);
end
if x(k,j)>range_x(k,2)
x(k,j)=range_x(k,2);
end
end
if f(x(:,j))<f(p(:,j))
p(:,j)=x(:,j);
end
end
best_value(i)=min(f(p));
if i>5&&abs(best_value(i)-best_value(i-5))<1e-5
break;
end
end
time=toc;
% disp(['Time used: ',num2str(time),' seconds'])
[mini,index]=min(f(p));
% disp(['fmin=',num2str(mini)]);
for k=1:n
% disp(['x',num2str(k),'=',num2str(p(k,index))]);
end
fprintf('%-30s | %-6.3f | %-6.3f | %-6.4f | %-6.4f | \n','PSO', p(k,index), mini, time, (mini-best_f)/best_f);