function [obj, grad] = compute_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,method,constant,exact,varargin)
% wrapper of computing objective and gradient
% 
%   [obj, grad] = compute_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,method,constant,varargin)
% 
% Input:
%   y : 2N x 1, vectorized 2-D coordinates
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
%   obj : objective
%   grad : gradient
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

switch method
    case 'linlog'
        lambda = varargin{1};
        if nargout>1
            [obj,grad] = compute_linlog_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,lambda);
        else
            obj = compute_linlog_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,lambda);
        end
    case 'ee'
        lambda = varargin{1};
        if nargout>1
            [obj,grad] = compute_ee_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,lambda);
        else
            obj = compute_ee_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,lambda);
        end
    case 'sne'
        if nargout>1
            [obj,grad] = compute_sne_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact);
        else
            obj = compute_sne_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact);
        end
    case 'ssne'
        if nargout>1
            [obj,grad] = compute_ssne_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact);
        else
            obj = compute_ssne_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact);
        end
    case 'tsne'
        if nargout>1
            [obj,grad] = compute_tsne_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact);
        else
            obj = compute_tsne_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact);
        end
    case 'wtsne'
        if nargout>1
            [obj,grad] = compute_wtsne_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact);
        else
            obj = compute_wtsne_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact);
        end
    case 'nerv'
        lambda = varargin{1};
        epsilon = varargin{2};
        if nargout>1
            [obj,grad] = compute_nerv_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,lambda,epsilon);
        else
            obj = compute_nerv_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,lambda,epsilon);
        end
    case 'mdsks'
        if nargout>1
            [obj,grad] = compute_mdsks_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact);
        else
            obj = compute_mdsks_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact);
        end        
    case 'gammane'
        gamma = varargin{1};
        kernel_type = varargin{2};
        if nargout>1
            [obj,grad] = compute_gammane_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,gamma,kernel_type);
        else
            obj = compute_gammane_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,constant,exact,gamma,kernel_type);
        end                
    otherwise
        error('unknown method');
end
