function [obj,grad] = compute_sne_obj_grad_repulsive_exact(Y)
n = size(Y,1);
q = exp(-distSqrdSelf(Y));
q(1:n+1:end) = 0;
obj = sum(log(sum(q,2)+eps));
if nargout>1
    Q = bsxfun(@rdivide, q, sum(q,2)+eps);
    Q = 0.5 * (Q+Q');
    grad = 4 * (Q-diag(sum(Q)))*Y;
end
