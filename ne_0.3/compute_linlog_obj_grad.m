function [obj,grad] = compute_linlog_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,lambda)
% compute LinLog objective and gradient
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

n = size(y,1) / dim;
Y = reshape(y, n, dim);

if exact
    [repu_obj,repu_grad] = compute_linlog_obj_grad_repulsive_exact(Y,lambda);
else
    if nargout==1
        repu_obj = compute_linlog_obj_grad_repulsive_barneshut(Y, theta, lambda, 1);
    else
        [repu_obj, repu_grad] = compute_linlog_obj_grad_repulsive_barneshut(Y, theta, lambda, 2);
    end
end

% myeps = 1e-50;
myeps = 0;
distnz = sqrt(sum((Y(I,:)-Y(J,:)).^2,2))+myeps;
obj = attr*sum(Pnz.*distnz) + repu_obj;
if nargout>1
    grad = attr * 4 * GraphLaplacian(sparse(I,J,0.5*Pnz./distnz,n,n)) * Y + repu_grad;
    grad = grad(:);
end
