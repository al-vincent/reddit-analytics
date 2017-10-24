function [obj,grad] = compute_ssne_obj_grad_repulsive_exact(Y)
n = size(Y,1);
q = exp(-distSqrdSelf(Y));
q(1:n+1:end) = 0;
obj = log(sum(sum(q))+eps);
if nargout>1
    Q = q./sum(sum(q));
    grad = 4 * (Q-diag(sum(Q)))*Y;
end
