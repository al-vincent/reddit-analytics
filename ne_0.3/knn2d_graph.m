function A = knn2d_graph(X, k)
% Calculate kNN graph of 2-D points in a fast way
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

[n,dim] = size(X);
if dim~=2
    error('This function only works for 2d points');
end

inds = cell(n,1);
ind = zeros(n*k,1);
tree = build_2dtree(X);
for i=1:n
    Xi = X;
    Xi(i,:) = nan;
    if k==1
        [inds{i}, ~] = nn_2dtree(tree, Xi, X(i,:));
    else
        [~,inds{i}] = knn_2dtree(tree, Xi, X(i,:), k);
    end
    ind((i-1)*k+1:i*k) = inds{i};
end

A = sparse(reshape(repmat(1:n, k,1), n*k,1), ind, ones(n*k,1), n, n, n*k);
