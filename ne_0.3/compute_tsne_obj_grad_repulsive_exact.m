function [obj,grad] = compute_tsne_obj_grad_repulsive_exact(Y)
n = size(Y,1);
q = 1./(1+distSqrdSelf(Y));
q(1:n+1:end) = 0;
obj = log(sum(sum(q))+eps);
if nargout>1
    Qq = q.*q./sum(sum(q));
    grad = 4 * (Qq-diag(sum(Qq)))*Y;
end
