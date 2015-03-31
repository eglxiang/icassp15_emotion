function [x] = IllinoisSolver(A,y,lambda,x0)
%Solve argmin_x : lambda||x||_1+||Ax-y||_2^2
%      argmin_x : ||x||_1+(1/lambda)||Ax-y||_2^2
%      argmin_x : ||x||_1+(beta/2)||Ax-y||_2^2 , beta = 2/lambda

At = A';
AtA = At*A ;
tau = eigs(AtA,1,'lm');
tauInv = 1/tau; % ADMM weight

beta=2/lambda; 
betaInv = 1/beta;
betaTauInv = betaInv*tauInv ;

temp=At*y;
tol_apg = 1e-6 ;
converged_apg = 0 ;
nIter_apg = 0 ;
maxIter_apg = 200 ;
x=x0;
t1 = 1 ; z = x;
while ~converged_apg

    nIter_apg = nIter_apg + 1 ;

    x_old_apg = x ;

    temp1 = z - tauInv*(AtA*z - temp) ;

    x = shrink(temp1, betaTauInv) ;

    if norm(x_old_apg - x) < tol_apg*norm(x_old_apg)
        converged_apg = 1 ;
    end

    if nIter_apg >= maxIter_apg
        converged_apg = 1 ;
    end

    t2 = (1+sqrt(1+4*t1*t1))/2 ;
    z = x + ((t1-1)/t2)*(x-x_old_apg) ;
    t1 = t2 ;
end
end

