function [A, G, repu_obj, quad_const] = compute_parts(Y,P,weights,Pnz,I,J,attr,theta,method,constant,exact,varargin)
% wrapper of computing parts for specific optimizers (e.g. 'mm' or 'fphssne')
% 
%   [A, G, repu_obj, quad_const] = compute_parts(Y,P,weights,Pnz,I,J,attr,theta,method,constant,varargin)
% 
% Input:
%   Y : N x dim, embedding coordinates
%   P : N x N, the pairwise similarities or network (weighted) adjacency
%   weights: weights of the data points (or graph nodes)
%   Pnz : = nonzeros(P)
%   I,J : the indices of non-zero entries in P
%   attr : the over-attraction scaling factor (default=1)
%   theta : the Barnes-Hut tree farness-factor parameter
%   method : the visualization method
%   constant : the constant part of the objective function
%   varargin : visualization method specific parameters
%              for LinLog, lambda = varargin{1}
%              for EE, lambda = varargin{1}
%              for NeRV, lambda = varargin{1}, epsilon = varargin{2}
%
% Output:
%   A : the coefficients of the quadratic part, i.e. W+W' in the paper
%   G : the gradient of the Lipschitz part (usually the repulsive part)
%   repu_obj: the repulsive objective
%   quad_const: the constant induced in the quadratification step
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

dim = size(Y,2);

n = size(P,1);
switch method
    case 'linlog'
        lambda = varargin{1};
%         myeps = 1e-50;
        myeps = 0;
        qnz = sqrt(sum((Y(I,:)-Y(J,:)).^2,2))+myeps;
        A = attr * sparse(I,J,0.5*Pnz./qnz,n,n);
        if exact
            [repu_obj, G] = compute_linlog_obj_grad_repulsive_exact(Y,lambda);
        else
            [repu_obj, G] = compute_linlog_obj_grad_repulsive_barneshut(Y, theta, lambda, 2);
        end
        quad_const = attr * 0.5*sum(Pnz.*qnz);
    case 'ee'
        lambda = varargin{1};
        A = attr * P;
        if exact
            [repu_obj, G] = compute_ee_obj_grad_repulsive_exact(Y, lambda);
        else
            [repu_obj, G] = compute_ee_obj_grad_repulsive_barneshut(Y, theta, lambda, 2);
        end
        quad_const = 0;
    case 'sne'
        A = attr * 0.5 * (P+P');
        if exact
            [repu_obj, G] = compute_sne_obj_grad_repulsive_exact(Y);
        else
            [repu_obj, G] = compute_sne_obj_grad_repulsive_barneshut(Y, theta, 2);
        end
        quad_const = 0;
    case 'ssne'
        A = attr * P;
        if exact
            [repu_obj, G] = compute_ssne_obj_grad_repulsive_exact(Y);
        else
            [repu_obj, G] = compute_ssne_obj_grad_repulsive_barneshut(Y, theta, 2);
        end
        quad_const = 0;
    case 'tsne'
        qnz = 1./(1+sum((Y(I,:)-Y(J,:)).^2,2));
        A = attr * sparse(I,J,Pnz.*qnz,n,n);
        if exact
            [repu_obj, G] = compute_tsne_obj_grad_repulsive_exact(Y);
        else
            [repu_obj, G] = compute_tsne_obj_grad_repulsive_barneshut(Y, theta, 2);
        end
        quad_const = attr * (sum(Pnz.*(qnz-log(qnz)-1)));
    case 'wtsne'
        qnz = 1./(1+sum((Y(I,:)-Y(J,:)).^2,2));
        A = attr * sparse(I,J,Pnz.*qnz,n,n);
        if exact
            [repu_obj, G] = compute_wtsne_obj_grad_repulsive_exact(Y, weights);
        else
            [repu_obj, G] = compute_wtsne_obj_grad_repulsive_barneshut(Y, weights, theta, 2);
        end
        quad_const = attr * (sum(Pnz.*(qnz-log(qnz)-1)));
    case 'nerv'
        lambda = varargin{1};
        epsilon = varargin{2};
        A = attr * 0.5 * (P+P');
        [obj, grad] = compute_nerv_obj_grad(Y(:),dim,P,weights,Pnz,I,J,attr,theta,constant,exact,lambda,epsilon);
        repu_obj = obj - trace(Y'*GraphLaplacian(P+P')*Y);
        G = reshape(grad, n,dim) - 4 * GraphLaplacian(A) * Y / attr;
        quad_const = 0;
    case 'mdsks'
        dist2nz = sum((Y(I,:)-Y(J,:)).^2,2);
        qnz = exp(-dist2nz);
        pqg = Pnz.*qnz;
        A = attr * sparse(I,J,pqg/sum(pqg),n,n);
        if exact
            [repu_obj, G] = compute_mdsks_obj_grad_repulsive_exact(Y);
        else
            [repu_obj, G] = compute_mdsks_obj_grad_repulsive_barneshut(Y, theta, 2);
        end
        qpq = qnz./sum(pqg);
        quad_const = attr * sum(Pnz.*(qpq.*log(qpq)));
    case 'gammane'
        gamma = varargin{1};
        kernel_type = varargin{2};
        if exact
            [repu_obj, G] = compute_gammane_repulsive_barneshut_exact(Y, gamma, kernel_type);
        else
            [repu_obj, G] = compute_gammane_repulsive_barneshut(Y, gamma, theta, kernel_type, 2);
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
        A = attr * sparse(I,J,pqg,n,n);
        if gamma<0
            quad_const = nan; % warning: should use mm1 instead of mm2
        else
            switch kernel_type
                case {1,2}
                    quad_const = attr * (sum(Pnz.*(qnz-log(qnz)-1)));
                otherwise
                    quad_const = 0;
            end
        end
    otherwise
        error('unknown method');
end
