rng('default');

%%
% ========================================================================
% The first demo illustrates how to use
% 1) x2p to get the similarity graph for small scale vectorial data
% 2) using the NeRV method and the l-BFGS optimizer
% 3) how to pass extra parameters to the visualization method
% ========================================================================
fprintf('First Demo: Iris data by NeRV\n');
load iris.mat;      % 'X': vectorial data (each row for a sample)
                    % 'C': class labels
                    % 'nc': number of classes
P = sparse(x2p(X));
method = 'nerv';
dim = 2;
attr = [];          % default value = 1
sphere = [];        % default value = false
theta = [];         % default value = 2
Y0 = [];            % default value = randn(size(P,1),2)*1e-4
max_iter = [];      % default value = 100
check_step = [];    % default value = 1/10 of max_iter
tol = [];           % default value = 1e-4
max_time = [];      % default value = 1 day
verbose = [];       % default value = true
optimizer = 'lbfgs';
recordYs = [];      % default value = false
exact = [];         % default value = false
lambda = 0.9;
epsilon = 1e-10;
Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer, recordYs, exact, lambda, epsilon);
DisplayVisualization(Y,C);
set(gcf, 'name', 'Iris data by NeRV (optimized by l-BFGS)', 'NumberTitle', 'off');


%% 
% ========================================================================
% The second demo illustrates how to use weighted t-SNE to visualize network data
% ========================================================================
clear;
fprintf('Second Demo: WorldTrade network by weighted t-SNE (optimized by MM) \n');
load WorldTrade_network.mat
Y = wtsne_p(A);
DisplayVisualization(Y,C);
set(gcf, 'name', 'WorldTrade network by weighted t-SNE (optimized by MM)', 'NumberTitle', 'off');


%%
% ========================================================================
% The third demo illustrates how to use
% 1) fast_approx_knn to get the similarity graph for large scale vectorial data
% 2) using the t-SNE method and the SD optimizer
% ========================================================================
clear;
fprintf('Third Demo: optdigits by t-SNE (optimized by Spectral Direction) \n');
load optdigits.mat;      % 'X': vectorial data (each row for a sample)
                         % 'C': class labels
                         % 'nc': number of classes
fprintf('Calculating 10NN...');
knn = fastknn(X, 10); % get 10NN graph
fprintf('done\n');
P = double(knn + knn' > 0);
method = 'tsne';
dim = 2;
attr = [];          % default value = 1
sphere = [];        % default value = false
theta = [];         % default value = 2
Y0 = [];            % default value = randn(size(P,1),2)*1e-4
max_iter = [];      % default value = 100
check_step = [];    % default value = 1/10 of max_iter
tol = [];           % default value = 1e-4
max_time = [];      % default value = 1 day
verbose = [];       % default value = true
optimizer = 'sd';
Y = ne_wrapper(P, method, dim, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, optimizer);
DisplayVisualization(Y,C);
set(gcf, 'name', 'Optdigits by t-SNE (optimized by SD)', 'NumberTitle', 'off');

%%
% ========================================================================
% The fourth demo illustrates how to use the shortcut functions 
% (e.g. tsne_p) for a given similarity graph
% using the default optimizer (MM)
% ========================================================================
clear;
fprintf('Fourth Demo: MNIST by t-SNE (optimized by MM) \n');
load mnist_70k_p.mat   % 'A': 10NN graph adjacency matrix
                       % 'C': class labels
                       % 'nc': number of classes
P = sparse(double(A+A'>0));
Y = tsne_p(P);
DisplayVisualization(Y,C);
set(gcf, 'name', 'MNIST by t-SNE (optimized by MM)', 'NumberTitle', 'off');
