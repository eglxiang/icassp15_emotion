function [x] = LassoSolver(A,y,lambda,x0,method_name)
addpath('lasso');
addpath('lasso/sub');

%Solve argmin_x : lambda||x||_1+||Ax-y||_2^2

if(strcmp(method_name,'IllinoisSolver'))
     x = IllinoisSolver(A,y,lambda,x0);
elseif(strcmp(method_name,'LassoGaussSeidel'))
     x = LassoGaussSeidel(A,y,lambda,'verbose',0);
elseif(strcmp(method_name,'LassoProjection'))
     x = LassoProjection(A,y,lambda,'verbose',0);
elseif(strcmp(method_name,'LassoGrafting'))     
     x =LassoGrafting(A,y,lambda,'verbose',0);
elseif(strcmp(method_name,'LassoShooting'))     
     x =LassoShooting(A,y,lambda,'verbose',0);
elseif(strcmp(method_name,'LassoBlockCoordinate'))     
     x =LassoBlockCoordinate(A,y,lambda,'verbose',0);
elseif(strcmp(method_name,'LassoSubGradient'))     
     x =LassoSubGradient(A,y,lambda,'verbose',0); 
else
    fprintf('Not Known Method \n');
end
lasso_value=lambda*sum(abs(x))+sum((A*x-y).*(A*x-y));
fprintf('LassoFunction = %f \n',lasso_value);     
end

