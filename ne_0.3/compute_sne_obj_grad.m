function [obj,grad] = compute_sne_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact)
% compute objective and gradient of Stochastic Neighbor Embedding (SNE)
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

n = size(y,1) / dim;
Y = reshape(y, n, dim);

if exact
    [repu_obj,repu_grad] = compute_sne_obj_grad_repulsive_exact(Y);
else
    if nargout==1
        repu_obj = compute_sne_obj_grad_repulsive_barneshut(Y, theta, 1);
    else
        [repu_obj, repu_grad] = compute_sne_obj_grad_repulsive_barneshut(Y, theta, 2);
    end
end

obj = constant + attr*sum(Pnz.*sum((Y(I,:)-Y(J,:)).^2,2)) + repu_obj;
if nargout>1
    grad = attr*2 * GraphLaplacian(P+P') * Y + repu_grad;
    grad = grad(:);
end
