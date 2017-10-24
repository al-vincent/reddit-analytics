function Y = wtsne_p(P, dim, sphere, do_init)
% compute 2-D coordinates by weighted t-SNE
%  
%   Y = wtsne_p(P, do_init)
% 
% Input: 
%   P: N x N, pairwise similarities or network (weighted) adjacency matrix
%   dim: dimensionality of the embedding (can be 2 or 3, default=2)
%   sphere: whether to use spherical embedding (default=false)
%   do_init: boolean, do over-attraction initialization or not, default=false
% 
% Output:
%   Y: N x 2, 2-D coordinates
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

if ~exist('dim', 'var') || isempty(dim)
    dim = 2;
end
if ~exist('sphere', 'var') || isempty(sphere)
    sphere = false;
end
if ~exist('do_init', 'var') || isempty(do_init)
    do_init = false;
end

method = 'wtsne';
P = preprocess_input_similarities((P+P')/2, method);

theta = 2;
rng('default');
Y0 = randn(size(P,1),dim)*1e-4;
check_step = 1;
tol = 1e-4;
max_time = inf;
verbose = true;
optimizer = 'mm';
recordYs = false;
exact = false;

if do_init
    fprintf('===============================================\n');
    fprintf('Initialization stage\n');
    fprintf('===============================================\n');
    attr = 4;
    max_iter = 30;
    Y1 = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs, exact);
else
    Y1 = Y0;
end

fprintf('===============================================\n');
fprintf('Long-run stage\n');
fprintf('===============================================\n');
attr = 1;
max_iter = 300;
Y = ne_wrapper(P, method, dim, Y1, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs, exact);
