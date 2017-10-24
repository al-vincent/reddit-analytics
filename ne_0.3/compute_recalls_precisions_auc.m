function auc = compute_recalls_precisions_auc(recalls, precisions)
% compute the area under the ROC curve (AUC) for given recalls and
% precisions
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

[recalls,ind] = sort(recalls);
precisions = precisions(ind);
n = length(recalls);

auc =sum((precisions(1:n-1) + precisions(2:end)) / 2 .* abs(recalls(1:n-1)-recalls(2:end)));
