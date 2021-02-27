%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kweihao@gmail.com 
% This code estimates the classification error of the best linear
% classifier in the logistic model.
% X: n*d matrix whose rows represent the n data points
% y: size n vector contains the labels of the n data points
% k: \beta^T\Sigma^i\beta for i = 2,3,...,k+1 are computed to estimate \beta^T\Sigma\beta
% lb,ub: lowerbound and upperbound of the eigenvalues of \Sigma.
% est: estimated classification error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function est = est_logistic(X,y,k,lb,ub)
    n = size(y,1);
    coef = compute_coef(lb,ub,k+1);
    G = triu(X*X',1);
    m=[];
    C = eye(n);
    for i = 1:k
        C = C*G;
        m(i)  = (y'*C*y)/nchoosek(n,(i+1));
    end
    bsb = m*coef;
    if bsb<0
        bsb=0;
    elseif bsb>1
        bsb=1;
    end
    est = 1/2-(sqrt(bsb)/2);
%    est = 1/2-ob_pred(sqrt(bsb)/2);   %ob_pred(t) is the function F_g(t) in the paper. 
end
function res_pred = ob_pred(res_ob)
    persistent pred ob;
    if isempty(pred)
        t = (0:0.01:100)';
        x = randn(1,100000);
        xp = x(x>0);
        pred = mean((1./(1+exp(-t*xp))-1/2),2);
        X = repmat(x,size(t));
        ob = mean(X./(1+exp(-t*x)),2);
    end 
    [~,I] = min(abs(ob-res_ob));
    res_pred = pred(I);
end
