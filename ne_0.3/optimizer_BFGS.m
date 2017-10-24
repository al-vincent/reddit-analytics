function [Y, Cs, ts, Ys, varargout] = optimizer_BFGS(P, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, method, recordYs, varargin)
% minimization by BFGS, using Matlab fminunc
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

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

options = optimset('OutputFcn', @record_time, ...
                   'LargeScale', 'off', ...
                   'MaxIter', max_iter, ...
                   'TolX', tol);

max_rec = 1e6;

Cs = nan(max_rec, 1); ts = nan(max_rec, 1);
Cs(1) = compute_obj_grad(Y(:),dim,P,weights,Pnz,I,J,attr,theta,method,constant,varargin{:});
ts(1) = 0;
if recordYs
    Ys = cell(max_iter,1);
    Ys{1} = Y;
else
    Ys = [];
end
t = 1;
yt = 0;
tic;

y = fminunc(@(y)compute_obj_grad(y,dim,P,weights,Pnz,I,J,attr,theta,method,constant,varargin{:}), Y(:), options);
Y = reshape(y, n, dim);

Cs = Cs(1:t);
ts = ts(1:t);
if recordYs
    Ys = Ys(1:yt);
end

    function stop = record_time(x, optimValues, state)
        stop = false;
        t = t + 1;
        ts(t) = toc;
        Cs(t) = optimValues.fval;
        
        if sphere
            tmp = spherify(reshape(x, n, dim));
            x = tmp(:);
        end
        
        if mod(optimValues.iteration,check_step)==0
            if recordYs
                yt = yt + 1;
                Ys{yt} = reshape(x, n, dim);
            end
            if verbose
                fprintf('t=%d, time=%6.6f, obj=%6.6f\n', t, ts(t), Cs(t));
            end
        end
        if ts(t)>max_time
            if verbose
                fprintf('max_time exceeded\n');
            end
            stop = true;
        end
    end
end
