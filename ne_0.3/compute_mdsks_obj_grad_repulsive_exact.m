function [obj,grad] = compute_mdsks_obj_grad_repulsive_exact(Y)
n = size(Y,1);
q = exp(-distSqrdSelf(Y));
qq = q.*q;
qq(1:n+1:end) = 0;
obj = 0.5*log(sum(sum(qq))+eps);
if nargout>1
    Q = qq./sum(sum(qq));
    grad = 4 * (Q-diag(sum(Q)))*Y;
end
