function tree = build_2dtree(points)
%
% depth even: split along y axis
%        odd: split along x axis
ids = 1:size(points,1);
tree = build_2dtree_recursive(points, ids, 1);


function tree = build_2dtree_recursive(points, ids, depth)
if isempty(points)
    tree = [];
else
    axis = 2-mod(depth,2);
%     [med,ind_med] = exact_median(points(:,axis)); % should be faster in theory, but actually slower in practice, maybe due to Matlab
    
    [val,ind_sorted] = sort(points(:,axis));
    med_pos = ceil(size(points,1)/2);
    med = val(med_pos);
    ind_med = ind_sorted(med_pos);
    tree.id = ids(ind_med);
    indLeft = points(:,axis)<med;
    tree.leftChild  = build_2dtree_recursive(points(indLeft,:), ids(indLeft), depth+1);
    indRight = points(:,axis)>med;
    tree.rightChild = build_2dtree_recursive(points(indRight,:), ids(indRight), depth+1);
end