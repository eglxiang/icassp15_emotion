function [L,A] = LR_group_sparse_rep(Y, D, group_label, lambdaG, lambdaL, eps,  maxIter,rho,tau)
% Optimization:  min_{A,L}: ||A||_1 + lambdaG*sum(||A_g||_F) + lambdaL*||L||_* st: Y = DA + L

[mD,nD] = size(D);
pY = size(Y,2);

% mu = 1e-6;
mu = 1e-4;
mu_max = 1e6;

A = zeros(nD,pY);
Z = zeros(mD,pY);
L = Z;

iter = 0;
converged = false;
stopCriterion = 1;

while ~converged       
    iter = iter + 1;
    
% Update L
    [U W V] = svd(Y - D*A + Z/mu,'econ'); 
%     [U W V] = lansvd(Y - D*A + Z/mu, maxR, 'L');
    
    diagW = diag(W);
    
    r_tem = length(find(diagW>lambdaL/mu));
    if r_tem>=1
        diagW = diagW(1:r_tem)-lambdaL/mu;
    else
        r_tem = 1;
        diagW = 0;
    end
    
    L = U(:,1:r_tem) * diag(diagW) * V(:,1:r_tem)';
    rkL(iter) = rank(L);

% Update A

    T = Y- L + Z/mu;
    G = D'*(D*A-T);
    R = A - tau*G;
    alpha = tau/mu;
    beta = lambdaG*tau/mu;
    
    for g = 1:length(group_label)
        idx = group_label{g};
        Rg = R(idx,:);
        
        H = max(Rg - alpha, 0) + min(Rg + alpha,0);
        nH = norm(H,'fro');
        
        if nH ==0;
             A(idx,:) = 0;
        else
            A(idx,:) =  H/nH*max(nH - beta, 0);
        end
    end 

   
% Update Z1, Z2    
    Z = Z + mu*(Y - L - D*A);
    
    mu = min(mu*rho, mu_max);

    stopCriterion = norm(Y-L-D*A, 'fro');
    
    if mod(iter,20) == 0
        display(['Iter ' num2str(iter) '     ' num2str(stopCriterion)]);
    end
    
    if stopCriterion < eps
        converged = true;
    end   
    
    if ~converged && iter >= maxIter
        converged = 1 ;       
    end
end
%figure, plot(rkL);
end