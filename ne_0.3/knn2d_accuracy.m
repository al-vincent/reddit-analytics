function acc = knn2d_accuracy(Y, C, nc, k)
% Calculate the kNN classification accuracy of 2-D points
% 
%   acc = knn2d_accuracy(Y, C, nc, k)
%
% Input:
%   Y : 2-D data points
%   C : ground truth class labels
%   nc : number of classes
%   k : numbero neighbors
% Output:
%   acc : kNN classification accuracy
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

A = knn2d_graph(Y, k);
dc = clabel2dataclasses(C, nc);
acc = trace(dc'*A*dc) / nnz(A);
