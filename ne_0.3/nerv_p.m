function Y = nerv_p(P,dim,sphere,lambda, epsilon)
% compute NeRV layout
% dim     - dimensionality of the embedding (can be 2 or 3, default=2)
% sphere  - whether to use spherical embedding (default=false)
% lambda  - controls the tradeoff; lambda=1 maximizes recall, lambda=0
%           maximizes precision, default lambda=0.9;
% epsilon - a remedy for zeros in P (avoid numerical problems in
%           precision, default=1e-10
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

if ~exist('dim', 'var') || isempty(dim)
    dim = 2;
end
if ~exist('sphere', 'var') || isempty(sphere)
    sphere = false;
end
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0.9;
end
if ~exist('epsilon', 'var') || isempty(epsilon)
    epsilon = 1e-10;
end

method = 'nerv';
P = preprocess_input_similarities((P+P')/2, method);

theta = 2;
rng('default');
Y0 = randn(size(P,1),dim)*1e-4;
check_step = 1;
tol = 1e-4;
max_time = inf;
verbose = true;
optimizer = 'sd';
recordYs = false;
exact = false;

fprintf('===============================================\n');
fprintf('Initialization stage\n');
fprintf('===============================================\n');
attr = 4;
max_iter = 30;
Y1 = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs, exact, lambda, epsilon);

fprintf('===============================================\n');
fprintf('Long-run stage\n');
fprintf('===============================================\n');
attr = 1;
max_iter = 100;
Y = ne_wrapper(P, method, dim, Y1, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs, exact, lambda, epsilon);
