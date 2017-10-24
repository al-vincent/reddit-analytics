function [obj,grad] = compute_linlog_obj_grad_repulsive_exact(Y,lambda)
n = size(Y,1);
dist2 = distSqrdSelf(Y)+eps;
dist2(1:n+1:end) = 1;
obj = -0.5*lambda*sum(sum(log(dist2)));
if nargout>1
    q = 1./(dist2+eps);
    q(1:n+1:end) = 0;
    grad = 2 * (q-diag(sum(q)))*Y;
end
