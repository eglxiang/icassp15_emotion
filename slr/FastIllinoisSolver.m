function [x]=FastIllinoisSolver(AtA,Atb,x0,tauInv,muTauInv,lasso_max_iter)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

tol_apg = 1e-6 ;
converged_apg = 0 ;
nIter_apg = 0 ;
%maxIter_apg = 50 ;
maxIter_apg=lasso_max_iter;
x=x0;
t1 = 1 ; z = x ;

while ~converged_apg

    nIter_apg = nIter_apg + 1 ;
    %fprintf('Lasso Iteration %d of %d \n',nIter_apg,lasso_max_iter);
    x_old_apg = x ;

    temp1 = z - tauInv*(AtA*z - Atb) ;

    x = shrink(temp1, muTauInv) ;

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

