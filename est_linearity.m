%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kweihao@gmail.com 
% This code estimates the square error of the best linear approximation of y
% X: n*d matrix whose rows represent the n data points
% y: size n vector contains the labels of the n data points
% k: \beta^T\Sigma^i\beta for i = 2,3,...,k+1 are computed to estimate \beta^T\Sigma\beta
% lb,ub: lowerbound and upperbound of the eigenvalues of \Sigma.
% est: estimated square error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function est = est_linearity(X,y,k,lb,ub)
    n = size(y,1);
    coef = compute_coef(lb,ub,k+1);
    G = triu(X*X',1);
    sigmay = mean(y.^2);
    m=[];
    C = eye(n);
    for i = 1:k
        C = C*G;
        m(i)  = (y'*C*y)/nchoosek(n,(i+1));
    end
    
    bsb = m*coef;
    est = (sigmay-bsb);
end 
