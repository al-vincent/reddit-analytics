function D2=distSqrd(X,Y)
% Calculate squard Euclidean distances between two sets of data points
% 
%   D2=distSqrd(X,Y)
%
% Input:
%   X : N x d
%   Y : M x d
% Output:
%   D2 : N x M
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

D2 = bsxfun(@plus, sum(X.^2,2), bsxfun(@minus, sum(Y.^2,2)', 2*(X*Y')));
