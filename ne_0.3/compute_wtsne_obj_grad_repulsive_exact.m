function [obj,grad] = compute_wtsne_obj_grad_repulsive_exact(Y,weights)
n = size(Y,1);
q = 1./(1+distSqrdSelf(Y));
q(1:n+1:end) = 0;
wq = (weights * weights') .* q;
obj = log(sum(sum(wq))+eps);
if nargout>1
    Qq = q.*wq./sum(sum(wq));
    grad = 4 * (Qq-diag(sum(Qq)))*Y;
end
