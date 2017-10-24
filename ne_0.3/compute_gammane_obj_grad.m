function [obj,grad] = compute_gammane_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,gamma,kernel_type)
n = size(y,1) / dim;
Y = reshape(y, n, dim);

if ~exist('exact', 'var') || isempty(exact)
    exact = false;
end

if exact
    [repu_obj, repu_grad] = compute_gammane_obj_grad_repulsive_exact(Y, gamma, kernel_type);
else
    if nargout==1
        repu_obj = compute_gammane_repulsive_barneshut(Y, gamma, theta, kernel_type, 1);
    else
        [repu_obj, repu_grad] = compute_gammane_repulsive_barneshut(Y, gamma, theta, kernel_type, 2);
    end
end

dist2nz = sum((Y(I,:)-Y(J,:)).^2,2);
switch kernel_type
    case 0
        qnz = exp(-dist2nz);
    case 1
        qnz = 1./(1+dist2nz);
    case 2
        qnz = 1./dist2nz;
end

switch gamma
    case -1
        obj = constant+attr*log(sum(Pnz./qnz))+repu_obj;
    case 0
        Pnzn = Pnz / sum(Pnz);
        obj = constant-attr*sum(Pnzn.*log(qnz))+repu_obj;
    case 1
        obj = constant-attr*log(sum(Pnz.*qnz))+repu_obj;
    otherwise
        obj = constant-attr*log(sum(Pnz.*qnz.^gamma))/gamma+repu_obj;
end

if nargout>1
    switch gamma
        case -1
            pqg = Pnz./qnz;
        case 0
            pqg = Pnz;
        case 1
            pqg = Pnz.*qnz;
        otherwise
            pqg = Pnz.*qnz.^gamma;
    end
    pqg = pqg / sum(pqg);
    switch kernel_type
        case {1,2}
            pqg = pqg .* qnz;
    end
    Lap_pqg = GraphLaplacian(sparse(I,J,pqg,n,n));
    grad = 4*attr*Lap_pqg*Y+repu_grad;
    grad = grad(:);
end