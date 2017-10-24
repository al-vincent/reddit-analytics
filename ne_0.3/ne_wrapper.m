function [Y, Cs, ts, Ys, extra] = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs, exact, varargin)
% Neighbor Embedding for Data Visualization
% 
%   Y = ne_wrapper(P)
%   Y = ne_wrapper(P, method)
%   Y = ne_wrapper(P, method, dim, Y0)
%   Y = ne_wrapper(P, method, dim, Y0, attr)
%   Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta)
%   Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step)
%   Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol)
%   Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time)
%   Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose)
%   Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer)
%   Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs)
%   Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs, exact)
%   Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs, exact, ...)
%   [Y, Cs, ts, Ys, extra] = ne_wrapper(...)
%
% Input
%   P          -   input similarities (n x n)
%   method     -   specifies the NE method; current avaiable choices
%                  'sne' (Stochastic Neighbor Embedding or SNE) 
%                  'ssne' (symmetric SNE, Gaussian kernel)
%                  'tsne' (t-Distributed SNE or t-SNE, default)
%                  'nerv' (Neighbor Retrieval Visualizer or NeRV)
%                  'wtsne' (weighted t-SNE)
%   dim        -   dimensionality of the embedding (default = 2)
%   Y0         -   initial mapped coordinates (n x dim)
%   attr       -   attraction factor (positive, default = 1)
%   sphere     -   whether to use 3D spherical embedding (default=false)
%   theta      -   controls Barnes-Hut approximation accuracy (larger is
%                  slower, but more accurate, default = 2)
%   max_iter   -   maximum number of iterations (default = 100)
%   check_step -   interval of checking (default = max_iter/10)
%   tol        -   tolerance of Y relative change (default = 1e-4)
%   verbose    -   display program output or not (default = true)
%   optimizer  -   optimization algorithm; current available options
%                  'sd'       (line search with spectral direction)
%                  'momentum' (steepest descent with momentum, fixed step size
%                  'gd'       (line search with steepest descent)
%                  'mm'       (Majorization-Minimization or MM, default)
%                  'fminunc'  (Matlab fminunc, basically BFGS)
%                  'lbfgs'    (l-BFGS algorithm)
%                  'vdm'      (the Barnes-Hut-SNE algorithm; basically same
%                              as momentum, but use over-attraction at 
%                              early iterations; only work for t-SNE)
%                  'fphssne'  (a fixed-point algorithm developed by Yang et
%                              al. in NIPS2009; only work for t-SNE)
%   recordYs   -   record Y at each check step or nor (default = false)
%   exact      -   use exact objective/gradient calculation or not (default
%                  = false, i.e. using Barnes-Hut Tree approximation)
%   varargin   -   additional parameters used by particular methods
%                  for LinLog, lambda = varargin{1}
%                  for EE, lambda = varargin{1}
%                  for NeRV, lambda = varargin{1}, epsilon = varargin{2}
%
% Output
%   Y          -   mapped coordinates (n x dim)
%   Cs         -   objectives
%   ts         -   elapsed time (in seconds)
%   Ys         -   mapped coordinates at each check step
%   extra      -   additional information for internal use
% The length of Cs, ts, Ys is max_iter/check_step + 1.
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

addpath(genpath('minFunc_2012'));
add_cmg_paths;

if ~exist('method', 'var') || isempty(method)
    method = 'tsne';
end

if ~exist('dim', 'var') || isempty(dim)
    dim = 2;
end

if ~exist('theta', 'var') || isempty(theta)
    theta = 2;
end

if ~exist('attr', 'var') || isempty(attr)
    attr = 1;
end

if ~exist('sphere', 'var') || isempty(sphere)
    sphere = false;
end

if ~exist('Y0', 'var') || isempty(Y0)
    Y0 = randn(size(P,1),dim)*1e-4;
end

if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 100;
end

if ~exist('check_step', 'var') || isempty(check_step)
    check_step = max(1,round(max_iter/10));
end

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-4;
end

if ~exist('verbose', 'var') || isempty(verbose)
    verbose = true;
end

if ~exist('optimizer', 'var') || isempty(optimizer)
    optimizer = 'mm';
end

if ~exist('recordYs', 'var') || isempty(recordYs)
    recordYs = false;
end

if ~exist('exact', 'var') || isempty(exact)
    exact = false;
end

if ~exist('max_time', 'var') || isempty(max_time)
    max_time = 60 * 60 * 24; % one day in seconds
end

switch optimizer
    case 'sd'
        optimizer_id = 1;
    case 'momentum'
        optimizer_id = 2;
    case 'gd'
        optimizer_id = 3;
    case 'mm'
        optimizer_id = 4;
    case 'fminunc'
        optimizer_id = 5;
    case 'lbfgs'
        optimizer_id = 6;
    case 'vdm'
        optimizer_id = 7;
    case 'fphssne'
        optimizer_id = 8;
    case 'mm0'
        optimizer_id = 9;
    case 'mm1'
        optimizer_id = 10;
    case 'mm2'
        optimizer_id = 11;
    case 'sd2'
        optimizer_id = 12;
    case 'sd3'
        optimizer_id = 13;
    otherwise
        error('unknown optimizer');
end

optimizer_functions = {...
    @optimizer_spectral_direction, ...
    @optimizer_momentum, ...
    @optimizer_gradient_descent, ...
    @optimizer_mm_backtrack_order2, ...
    @optimizer_BFGS, ...
    @optimizer_minFunc, ...
    @optimizer_vdm, ...
    @optimizer_fphssne, ...
    @optimizer_mm_backtrack_order0, ...
    @optimizer_mm_backtrack_order1, ...
    @optimizer_mm_backtrack_order2, ...
    @optimizer_spectral_direction2, ...
    @optimizer_spectral_direction3, ...
    };

optimizer_function = optimizer_functions{optimizer_id};
[Y, Cs, ts, Ys, extra] = optimizer_function(P, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, method, recordYs, exact, varargin{:});
