function Y = sne_p(P,dim,sphere)
% compute Stochastic Neighbor Embedding (SNE) layout
% dim     - dimensionality of the embedding (can be 2 or 3, default=2)
% sphere  - whether to use spherical embedding (default=false)
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

if ~exist('dim', 'var') || isempty(dim)
    dim = 2;
end

if ~exist('sphere', 'var') || isempty(sphere)
    sphere = false;
end

method = 'sne';
P = preprocess_input_similarities(P, method);

theta = 2;
rng('default');
Y0 = randn(size(P,1),dim)*1e-4;
check_step = 1;
tol = 1e-4;
max_time = inf;
verbose = true;
optimizer = 'mm';
recordYs = false;

fprintf('===============================================\n');
fprintf('Initialization stage\n');
fprintf('===============================================\n');
attr = 4;
max_iter = 30;
Y1 = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs);

fprintf('===============================================\n');
fprintf('Long-run stage\n');
fprintf('===============================================\n');
attr = 1;
max_iter = 100;
Y = ne_wrapper(P, method, dim, Y1, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs);

