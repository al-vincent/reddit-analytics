function L = GraphLaplacian(A)
% Calculate the Laplacian of a given graph
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

L = diag(sum(A)) - A;
