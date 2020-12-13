clear all
clc
%% Initialization

syms X Y lambda
i = 1;
var = 2;
eps = 0.01;
x = [0;0];
f = symfun(X - Y + 2*X^2 + 2*X*Y + Y^2, [X Y]);
S = eye(var);

%% Looping

for i = 1:20    
    x_temp = x(:,i);
    if mod(i,2) == 1
        s = S(:,1);
    else
        s = S(:,2);
    end
    
    fval = f(x_temp(1), x_temp(2));
    xp_temp = x_temp + eps*s;
    xn_temp = x_temp - eps*s;
    fval_p = f(xp_temp(1), xp_temp(2));
    fval_n = f(xn_temp(1), xn_temp(2));
    
    if fval_p < fval
        s = s;
    elseif fval_n < fval
        s = -s;
    else
        break
    end
    
    xopt_temp = x_temp + lambda*s;
    fval_opt = f(xopt_temp(1), xopt_temp(2));
    grad_fopt = gradient(fval_opt);
    lambda_opt = solve(grad_fopt);
    
    x(:,i+1) = x_temp + lambda_opt*s;
end

%% Plotting

axis square
X = linspace(-5,5);             Y = linspace(-5,5);
[A,B] = meshgrid(X,Y);
f_fig = f(A,B);
levels = 10:10:350;
figure(1), contour(X,Y,f_fig,levels,'linewidth',1.2), colorbar
hold on;
plot(x(1,:), x(2,:))