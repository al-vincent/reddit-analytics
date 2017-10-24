function [id, dist2] = nn_2dtree(tree, points, query)
% Calculate the nearest distance and index of 2-D points for a given query point
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

[dist2, id] = nn_2dtree_recursive(tree, points, query, 1, Inf);

function [NNDist2, NNId] = nn_2dtree_recursive(tree, points, query, depth, currentNNDist2)
if isempty(tree)
    NNDist2 = Inf;
    NNId = [];
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
[NNDist2, NNId] = nn_2dtree_recursive(nearer_kd, points, query, depth+1, currentNNDist2);
currentNNDist2 = min(currentNNDist2, NNDist2);

if sqrt(currentNNDist2)>=abs(dq)
    dist2c = sum((loc-query).^2);
    if dist2c<NNDist2
        NNId = tree.id;
        NNDist2 = dist2c;
        currentNNDist2 = NNDist2;
    end
    [NNDist2_tmp, NNId_tmp] = nn_2dtree_recursive(further_kd, points, query, depth+1, currentNNDist2);
    if NNDist2_tmp<NNDist2
        NNId = NNId_tmp;
        NNDist2 = NNDist2_tmp;
    end
end
