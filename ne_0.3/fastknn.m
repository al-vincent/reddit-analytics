function knn = fastknn(X, k, verbose)
% Calculate kNN in a fast way
% A Matlab wrapper that calls the ball tree implementation
% 
%   knn = fastknn(X, k)
%   knn = fastknn(X, k, verbose)
%
% Input:
%   X : N x d
%   k : the number of neighbors
%   verbose : to display the progress or not, default=false
% Output:
%   knn : N x N
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

if ~exist('verbose', 'var') || isempty(verbose)
    verbose = false;
end

n = size(X,1);

X = X';

J = reshape(repmat(1:n, k, 1), n*k, 1);
I = fastknn_mex(X, k, verbose);
knn = sparse(I, J, ones(n*k,1), n, n, n*k)';
