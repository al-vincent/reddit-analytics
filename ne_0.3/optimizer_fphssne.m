function [Y, Cs, ts, Ys, varargout] = optimizer_fphssne(P, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, method, recordYs, varargin)
% minimization by the Fixed-Point Heavy-tailed Symmetric SNE (FPHSSNE)
% algorithm (Zhirong Yang et al. in NIPS2009)
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

% currtently only suitable for tsne
varargout{1} = [];
n = size(P,1);
P = preprocess_input_similarities(P,method);

Y = Y0;
dim = size(Y,2);

if sphere
    Y = spherify(Y);
end

[I,J] = find(P);
Pnz = nonzeros(P);
weights = get_weights(P);
constant = get_constant_term(P,weights,Pnz,I,J,method,varargin{:});

Cs = nan(max_iter, 1); ts = nan(max_iter, 1);
Cs(1) = compute_obj_grad(Y(:),dim,P,weights,Pnz,I,J,attr,theta,method,constant,varargin{:});
ts(1) = 0;
t = 1;
if recordYs
    Ys = cell(max_iter,1);
    Ys{1} = Y;
else
    Ys = [];
end
tic;
for iter=2:max_iter
    Y_old = Y;
    
    [Pq, repu] = compute_parts(Y,P,weights,Pnz,I,J,attr,theta,method,constant,varargin{:});
    Y = bsxfun(@rdivide, Pq * Y - repu/4, sum(Pq,2)+eps);

    if sphere
        Y = spherify(Y);
    end
    
    obj = compute_obj_grad(Y(:),dim,P,weights,Pnz,I,J,attr,theta,method,constant,varargin{:});
    
    if recordYs && mod(iter, check_step)==0
        t = t + 1;
        Ys{t} = Y;
    end
    
    ts(iter) = toc;
    Cs(iter) = obj;
    
    if ts(iter)>max_time
        if verbose
            fprintf('max_time exceeded\n');
        end
        break;
    end
    
    diffY = norm(Y-Y_old,'fro') / norm(Y_old,'fro');
    if verbose && mod(iter, check_step)==0
        fprintf('iter=%d/%d, collapsed_time=%.2f, diffY=%.12f, obj=%.12f\n', iter, max_iter, toc, diffY, obj);
    end
    
    if diffY<tol
        if verbose
            fprintf('converged after %d iterations.\n', iter);
        end
        break;
    end
end

Cs = Cs(1:iter);
ts = ts(1:iter);
if recordYs
    Ys = Ys(1:t);
end
