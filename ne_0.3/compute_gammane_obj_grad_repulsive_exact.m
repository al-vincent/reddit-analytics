function [repu_obj, repu_grad] = compute_gammane_obj_grad_repulsive_exact(Y, gamma, kernel_type)
n = size(Y,1);
dist2 = distSqrdSelf(Y);
switch kernel_type
    case 0
        q = exp(-dist2);
    case 1
        q = 1./(1+dist2);
    case 2
        q = 1./dist2;
end
q(1:n+1:end) = 0;

switch gamma
    case -1
        repu_obj = sum(ln(nonzeros(q)))/(n*(n-1));
    case 0
        repu_obj = log(sum(sum(q)));
    otherwise
        repu_obj = log(sum(sum(q.^(1+gamma))))/(1+gamma);
end

if nargout>1
    switch gamma
        case -1
            Q = ones(n) / (n*(n-1));
        case 0 
            Q = q / sum(sum(q));
        otherwise
            Q = q.^(1+gamma);
            Q = Q / sum(sum(Q));
    end
    switch kernel_type
        case {1,2}
            Q = Q .* q;
    end
    repu_grad = -4 * GraphLaplacian(Q) * Y;
end
