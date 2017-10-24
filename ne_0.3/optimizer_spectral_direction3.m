function [Y, Cs, ts, Ys, varargout] = optimizer_spectral_direction3(P, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, method, recordYs, exact, varargin)
% minimization using the spectral direction (Vladymyrov and
% Carreira-Perpinan in ICML2012). Use cmg_sdd.
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.


varargout{1} = zeros(2,1);
n = size(P,1);
P = preprocess_input_similarities(P,method);

Y = Y0;
dim = size(Y,2);

if sphere
    Y = spherify(Y);
end

base_eta = .6;                                                          % initial step size
progTol = 1e-9;                                                        % minimum allowable step size
c1 = 1e-4;                                                               % Armijo sufficient decrease parameter
c2 = .9;                                                                % curvature parameter
max_search = 25;                                                        % maximum number of function evaluations
LS_interp = 2;
LS_multi = 0;

[I,J] = find(P);
Pnz = nonzeros(P);
weights = get_weights(P);
constant = get_constant_term(P,weights,Pnz,I,J,method,varargin{:});
LP = get_LP(P,method);

opts.display=0;
pfunLP = cmg_sdd(LP,opts);

warning('off', 'MATLAB:nearlySingularMatrix');

Cs = nan(max_iter, 1); ts = nan(max_iter, 1);
Cs(1) = compute_obj_grad(Y(:),dim,P,weights,Pnz,I,J,attr,theta,method,constant,exact,varargin{:});
ts(1) = 0;
t = 1;
if recordYs
    Ys = cell(max_iter,1);
    Ys{1} = Y;
else
    Ys = [];
end
total_eval = 0;
tic;
for iter=2:max_iter
    Y_old = Y;
    
    switch method
        case {'tsne', 'wtsne'}
            [M, repu, repu_obj, ~] = compute_parts(Y,P,weights,Pnz,I,J,attr,theta,method,constant,exact,varargin{:});
            LM = diag(sum(M))-M;
            C = constant + attr*sum(Pnz.*log(1+sum((Y(I,:)-Y(J,:)).^2,2))) + repu_obj;
            dC = 4 * LM * Y + repu;
            dC = dC(:);
        otherwise
            LM = LP;
            [C,dC] = compute_obj_grad(Y(:),dim,P,weights,Pnz,I,J,attr,theta,method,constant,exact,varargin{:});
    end
    LM = LM + 1e-10 * min(diag(LM)) * speye(size(P,1));
    switch method
        case {'tsne', 'wtsne'}
            pfun = cmg_sdd(LM,opts);
        otherwise
            pfun = pfunLP;
    end
    
    grad = reshape(dC, size(Y));
    if iter>2
        direction_old = reshape(direction, n, dim);
    else
        direction_old = zeros(n,dim);
    end
    if exact
        direction = -LM\grad;
    else
        direction = zeros(n,dim);
        for d=1:dim;
            direction(:,d) = ConjugateGradientSolver(LM, -grad(:,d),direction_old(:,d),pfun);
%             direction(:,d) = ConjugateGradientSolver(LM, -grad(:,d), direction_old(:));
        end
    end
    direction = direction(:);
        
    gtd = dC' * direction;
    [eta, C, ~, no_eval] = WolfeLineSearch(Y(:), base_eta, direction, C, dC, ...
        gtd, c1, c2, LS_interp, LS_multi, max_search, progTol, false, false, false, ...
        @compute_obj_grad, dim,P,weights,Pnz,I,J,attr,theta,method,constant,exact,varargin{:});
    
    total_eval = total_eval + no_eval;
    Y = Y + eta * reshape(direction, size(Y));
    Y = bsxfun(@minus, Y, mean(Y));

    if sphere
        Y = spherify(Y);
    end

    if recordYs && mod(iter, check_step)==0
        t = t + 1;
        Ys{t} = Y;
    end
    
    Cs(iter) = C;
    ts(iter) = toc;
    
    if ts(iter)>max_time
        if verbose
            fprintf('max_time exceeded\n');
        end
        break;
    end
    
    if verbose && mod(iter, check_step)==0
        fprintf('iter=%d, obj=%f, eta=%.12f, %d objfun calls, %.2f seconds used\n', iter, C, eta, no_eval, ts(iter));
    end
    
    if norm(Y-Y_old,'fro')/norm(Y_old,'fro')<tol
        if verbose
            fprintf('converged after %d iterations.\n', iter);
        end
        break;
    end
end
varargout{1} = [total_eval, iter];

Cs = Cs(1:iter);
ts = ts(1:iter);
if recordYs
    Ys = Ys(1:t);
end
