function Y = linlog(P,dim,lambda)
% compute LinLog layout
% dim      - dimensionality of the embedding (can be 2 or 3, default=2)
% lambda   - tradeoff between attraction and repulsion, default lambda=1
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.


if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1;
end
if ~exist('dim', 'var') || isempty(dim)
    dim = 2;
end

method = 'linlog';
P = preprocess_input_similarities((P+P')/2, method);

theta = 10;
rng('default');
Y0 = randn(size(P,1),dim)*1e4;
check_step = 1;
tol = 1e-4;
max_time = inf;
verbose = true;
optimizer = 'mm';

% optimizer = 'sd';
%optimizer = 'mm'; % this is currently numerically problematic because LM 
%                    can be close to indefinite such that 
%                    ConjugateGradientSolver fails to give the correct minimum


fprintf('===============================================\n');
fprintf('Initialization stage\n');
fprintf('===============================================\n');
attr = 4;
max_iter = 30;
Y1 = ne_wrapper(P, method, dim, Y0, attr, theta, max_iter, check_step, tol, max_time, verbose, optimizer, lambda);

fprintf('===============================================\n');
fprintf('Long-run stage\n');
fprintf('===============================================\n');
attr = 1;
max_iter = 100;
Y = ne_wrapper(P, method, dim, Y1, attr, theta, max_iter, check_step, tol, max_time, verbose, optimizer, lambda);
