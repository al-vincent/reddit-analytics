function [Y, Cs, ts, Ys, varargout] = optimizer_mm_backtrack_various_order(P, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, method, recordYs, exact, order, varargin)
% minimization by Majorization-Minimization, with a given order of surrogate
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.


varargout{1} = zeros(2,1); % 1st element: total backtrack times; 2nd element: total iterations used
% note: varargout{2} is record of L values
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
% weights = ones(n,1)/n;
constant = get_constant_term(P,weights,Pnz,I,J,method,varargin{:});

L = 1e-6;
max_try = 30;
% max_try = 300;

Cs = nan(max_iter, 1); ts = nan(max_iter, 1);
Cs(1) = compute_obj_grad(Y(:),dim,P,weights,Pnz,I,J,attr,theta,method,constant,exact,varargin{:});
obj = Cs(1);
ts(1) = 0;
t = 1;
if recordYs
    Ys = cell(max_iter,1);
    Ys{1} = Y;
else
    Ys = [];
end

Ls = nan(max_iter,1);
Ls(1) = L;

total_try = 0;
tic;
for iter=2:max_iter
    Y_old = Y;
    
    [M, repu, repu_obj, quad_const] = compute_parts(Y,P,weights,Pnz,I,J,attr,theta,method,constant,exact,varargin{:});
    LM = diag(sum(M))-M;
    grad = 4 * LM * Y + repu;
    
    Ltry = L / 2;
    Ytry = Y;
    ntry = 0;
    while ntry<max_try
        A = 4 * LM + Ltry * speye(n);
        B = -repu + Ltry * Y;
        if exact
            Ytry = A\B;
        else
            for d=1:size(B,2)
                opts.display=0;
                pfun = cmg_sdd(A,opts);
                Ytry(:,d) = ConjugateGradientSolver(A, B(:,d),Y(:,d),pfun);
            end
        end
        obj_try = compute_obj_grad(Ytry(:),dim,P,weights,Pnz,I,J,attr,theta,method,constant,exact,varargin{:});
        switch order
            case 0
                surrogate_obj = obj;
            case 1
                surrogate_obj = obj + trace((Ytry-Y)'*grad) + 0.5*Ltry*norm(Ytry-Y,'fro').^2;
            case 2
                surrogate_obj = 2*trace(Ytry'*LM*Ytry) + quad_const ...
                    + repu_obj + trace((Ytry-Y)'*repu) + 0.5*Ltry*norm(Ytry-Y,'fro').^2 ...
                    + constant;
        end
        
        ntry = ntry + 1;
        if obj_try<surrogate_obj && obj_try<obj % the latter condition is needed when LM is close to indefinite
            L = Ltry;
            Y = Ytry;
            obj = obj_try;
            break;
        end
        Ltry = Ltry * 2;
    end
    
    if sphere
        Y = spherify(Y);
    end
    
    Ls(iter) = L;
    
    total_try = total_try + ntry;
    if ntry>=max_try
        if verbose
            fprintf('max ntry in backtrack at the %d iteration.\n', iter);
        end
        break;
    end
    
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
        fprintf('iter=%d/%d, L=%.20f, ntry=%d, elapsed_time=%.2f, diffY=%.12f, obj=%.12f\n', iter, max_iter, L, ntry, toc, diffY, obj);
    end
    
    if diffY<tol
        if verbose
            fprintf('converged after %d iterations.\n', iter);
        end
        break;
    end
end
varargout{1} = [total_try, iter];
varargout{2} = Ls(~isnan(Ls));

indval = ~isnan(ts);
Cs = Cs(indval);
ts = ts(indval);
% Cs = Cs(1:iter);
% ts = ts(1:iter);
if recordYs
    Ys = Ys(1:t);
end

