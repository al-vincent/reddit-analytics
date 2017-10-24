function [obj, grad] = compute_nerv_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,lambda,epsilon)
% compute objective and gradient of Neighbor Retrieval Visualizer (NeRV)
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

n = size(y,1) / dim;
Y = reshape(y, n, dim);
if exact
    [obj,grad] = compute_nerv_obj_grad_exact(Y,P,Pnz,I,J,lambda,epsilon);
else
    if nargout==1
        obj = compute_nerv_obj_grad_barneshut(Y, P', lambda, epsilon, theta, 1);
    else
        [obj, grad] = compute_nerv_obj_grad_barneshut(Y, P', lambda, epsilon, theta, 2);
        grad = grad(:);
    end
end
if attr~=1
    obj = obj + (attr-1) * trace(Y'*GraphLaplacian(P+P')*Y);
    if nargout>1
        tmp = (attr-1) * 2 * GraphLaplacian(P+P') * Y;
        grad = grad + tmp(:);
    end
%     obj = obj + (attr-1) * 2 * trace(Y'*GraphLaplacian(P+P')*Y);
%     grad = grad + (attr-1) * 4 * GraphLaplacian(P+P') * Y;
end
