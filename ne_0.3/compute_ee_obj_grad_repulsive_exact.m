function [obj,grad] = compute_ee_obj_grad_repulsive_exact(Y,lambda)
n = size(Y,1);
q = exp(-distSqrdSelf(Y));
q(1:n+1:end) = 0;
obj = lambda*sum(sum(q,2));
if nargout>1
    grad = lambda * 4 * (q-diag(sum(q)))*Y;
end
