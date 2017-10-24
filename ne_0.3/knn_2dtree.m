function [kNNDist2s, kNNIds] = knn_2dtree(tree, points, query, k)
% Calculate kNN distances and indices of 2-D points for a given query point
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

kNNDist2s = ones(k,1)*Inf;
kNNIds = zeros(k,1);
[kNNDist2s, kNNIds] = knn_2dtree_recursive(tree, points, query, k, 1, kNNDist2s, kNNIds);

function [kNNDist2s, kNNIds] = knn_2dtree_recursive(tree, points, query, k, depth, currentkNNDist2s, currentkNNIds)
if isempty(tree)
    kNNDist2s = ones(k,1)*Inf;
    kNNIds = zeros(k,1);
    return;
end
if mod(depth,2)==0
    dim = 2;
else
    dim = 1;
end
loc = points(tree.id,:);
dq = query(dim)-loc(dim);
if dq<0
    nearer_kd = tree.leftChild;
    further_kd = tree.rightChild;
else
    nearer_kd = tree.rightChild;
    further_kd = tree.leftChild;
end
[kNNDist2s, kNNIds] = knn_2dtree_recursive(nearer_kd, points, query, k, depth+1, currentkNNDist2s, currentkNNIds);

if sqrt(kNNDist2s(end))>=abs(dq)
    dist2c = sum((loc-query).^2);
    if dist2c<kNNDist2s(end)
        i = k;
        while i>0 && dist2c<kNNDist2s(i)
            i = i - 1;
        end
        if i==k-1
            kNNIds(k) = tree.id;
            kNNDist2s(k) = dist2c;
        else
            kNNIds((i+2):end) = kNNIds((i+1):(end-1));
            kNNIds(i+1) = tree.id;
            kNNDist2s((i+2):end) = kNNDist2s((i+1):(end-1));
            kNNDist2s(i+1) = dist2c;
        end
        currentkNNDist2s = kNNDist2s;
        currentkNNIds = kNNIds;
    end
    [kNNDist2s_tmp, kNNIds_tmp] = knn_2dtree_recursive(further_kd, points, query, k, depth+1, currentkNNDist2s, currentkNNIds);
    if kNNDist2s_tmp(1)<kNNDist2s(end)
        [val, ind] = sort([kNNDist2s; kNNDist2s_tmp]);
        kNNDist2s = val(1:k);
        ids = [kNNIds; kNNIds_tmp];
        kNNIds = ids(ind(1:k));
    end
end

