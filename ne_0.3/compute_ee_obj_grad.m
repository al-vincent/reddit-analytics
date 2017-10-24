function [obj,grad] = compute_ee_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,lambda)
% compute Elastic Embedding (EE) objective and gradient
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

n = size(y,1) / dim;
Y = reshape(y, n, dim);

if exact
    [repu_obj,repu_grad] = compute_ee_obj_grad_repulsive_exact(Y,lambda);
else
    if nargout==1
        repu_obj = compute_ee_obj_grad_repulsive_barneshut(Y, theta, lambda, 1);
    else
        [repu_obj, repu_grad] = compute_ee_obj_grad_repulsive_barneshut(Y, theta, lambda, 2);
    end
end

obj = attr*sum(Pnz.*sum((Y(I,:)-Y(J,:)).^2,2)) + repu_obj;
if nargout>1
    grad = attr * 4 * GraphLaplacian(P) * Y + repu_grad;
    grad = grad(:);
end
