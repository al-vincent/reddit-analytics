function dist2 = distSqrdSelf(X)
% Calculate pairwise squard Euclidean distances of a set of data points
% 
%   dist2=distSqrdSelf(X)
%
% Input:
%   X : N x d
% Output:
%   dist2 : N x N
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.


sx = sum(X .^ 2, 2);
dist2 = abs(bsxfun(@plus, sx, bsxfun(@plus, sx', -2 * X * X')));
