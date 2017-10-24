function [Y, Cs, ts, Ys, varargout] = optimizer_vdm(P, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, method, recordYs, varargin)
% minimization using the van der Maaten and Hinton's algorithm; parameter
% setting follow the bh_tsne code
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.


% van der Maarten's original algorithm
varargout{1} = [];
attr = 1;
n = size(P,1);
P = preprocess_input_similarities(P,method);

Y = Y0;
dim = size(Y,2);

if sphere
    Y = spherify(Y);
end

%%% parameter values used in Simple_tSNE
% momentum = 0.5;                                     % initial momentum
% final_momentum = 0.8;                               % value to which momentum is changed
% mom_switch_iter = 250;                              % iteration at which momentum is changed
% epsilon = 500;                                      % initial learning rate
% min_gain = .01;
% stop_lying_iter = 100;

%%% parameter values used in bh_tsne
momentum = 0.5;                                     % initial momentum
final_momentum = 0.8;                               % value to which momentum is changed
mom_switch_iter = 250;                              % iteration at which momentum is changed
epsilon = 200;                                      % initial learning rate
min_gain = .01;
stop_lying_iter = 250;



if n<10000;
    lying_factor = 4;
else
    lying_factor = 12;
end
P = P * lying_factor;		% lie about the P-vals

y_incs  = zeros(size(Y));
gains = ones(size(Y));

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
bNaN = false;
tic;
for iter=2:max_iter
    Y_old = Y;
    
    [~,dC] = compute_obj_grad(Y(:),dim,P,weights,Pnz,I,J,attr,theta,method,constant,varargin{:});

    y_grads = reshape(dC, size(Y));
    
    gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...            % note that the y_grads are actually -y_grads
        + (gains * .8) .* (sign(y_grads) == sign(y_incs));
    gains(gains < min_gain) = min_gain;
    y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
    Ytry = Y + y_incs;
    if nnz(isnan(Ytry))>0 || nnz(isinf(Ytry))>0
        if verbose
            fprintf('NaN or Inf in Ytry!!\n');
        end
        bNaN = true;
        break;
    end
    
    Y = bsxfun(@minus, Ytry, mean(Ytry, 1));
    
    if sphere
        Y = spherify(Y);
    end
    
    if iter == mom_switch_iter
        momentum = final_momentum;
    end
    if iter == stop_lying_iter
        P = P ./ lying_factor;
        Pnz = nonzeros(P);
        weights = get_weights(P);
        constant = get_constant_term(P,weights,Pnz,I,J,method);
    end
    
    if recordYs && mod(iter, check_step)==0
        t = t + 1;
        Ys{t} = Y;
    end
    
    if iter>=stop_lying_iter
        [Cs(iter),~] = compute_obj_grad(Y(:),dim,P,weights,Pnz,I,J,attr,theta,method,constant,varargin{:});
    else
        P_true = P./lying_factor;
        Pnz_true = nonzeros(P_true);
        weights_true = weights./lying_factor;
        constant_true = get_constant_term(P_true,weights_true,Pnz_true,I,J,method);
        [Cs(iter),~] = compute_obj_grad(Y(:),dim,P_true,weights_true,Pnz_true,I,J,attr,theta,method,constant_true,varargin{:});
    end
    ts(iter) = toc;
    
    if ts(iter)>max_time
        if verbose
            fprintf('max_time exceeded\n');
        end
        break;
    end
    
    if verbose && mod(iter, check_step)==0
        fprintf('iter=%d, %.2f seconds used\n', iter, ts(iter));
    end
    
    if norm(Y-Y_old,'fro')/norm(Y_old,'fro')<1e-6
        if verbose
            fprintf('converged after %d iterations.\n', iter);
        end
        break;
    end
end
if bNaN
    iter = iter - 1; % discard the last iteration result
end
Cs = Cs(1:iter);
ts = ts(1:iter);
if recordYs
    if nnz(isnan(Ys{t}))>0
        t = t - 1;
    end
    Ys = Ys(1:t);
end
