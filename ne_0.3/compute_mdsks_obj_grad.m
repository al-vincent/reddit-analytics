function [obj,grad] = compute_mdsks_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact)
% compute objective and gradient of MDS based on kernel similarity (MDS-KS)
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

n = size(y,1) / dim;
Y = reshape(y, n, dim);

if exact
    [repu_obj,repu_grad] = compute_mdsks_obj_grad_repulsive_exact(Y);
else
    if nargout==1
        repu_obj = compute_mdsks_obj_grad_repulsive_barneshut(Y, theta, 1);
    else
        [repu_obj, repu_grad] = compute_mdsks_obj_grad_repulsive_barneshut(Y, theta, 2);
    end
end

qnz = exp(-sum((Y(I,:)-Y(J,:)).^2,2));
obj = constant - attr * log(sum(Pnz.*qnz)) + repu_obj;

if nargout>1
    pqg = Pnz.*qnz;
    Lap_pqg = GraphLaplacian(sparse(I,J,pqg/sum(pqg),n,n));
    grad = 4*attr*Lap_pqg*Y+repu_grad;
    grad = grad(:);
end
