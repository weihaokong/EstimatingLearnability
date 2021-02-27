function [coef2,max_allowed] = compute_coef(lb,ub,k)
    eps = (ub-lb)/1000;
    x = (lb:eps:ub)';
    if (isempty(x))
        x=lb;
    end
    A=[];
    for t=2:k
        A = [A x.^t];
    end
    %% compute l_infty error
    newA = [A -1*ones(size(x));-A -1*ones(size(x))];    
    b = [x;-x];
    f = [zeros(k-1,1);1;];
    [coef,err] = linprog(f,newA,b);
    coef = coef(1:end-1);
    
    %% [a_2...a_k abs(a_2)...abs(a_k) t]
    max_allowed = err*2;
    weight = 2.^(1:(k-1))';
    newA2 = [A zeros(size(A)) -1*ones(size(x));...
        -A zeros(size(A)) -1*ones(size(x)) ; ...
        eye(k-1) -1*eye(k-1) zeros(k-1,1);...
        -eye(k-1) -1*eye(k-1) zeros(k-1,1);...
        zeros(1,k-1) zeros(1,k-1) 1];    
    b2 = [x;-x;zeros(2*(k-1),1);max_allowed];
    f2 = [zeros(k-1,1);weight;0];
    [coef2,err2] = linprog(f2,newA2,b2);
    coef2 = coef2(1:k-1);
end