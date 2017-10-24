function auc = compute_visualization_auc(Y,C)
% compute the area under the ROC curve (AUC) for given 2-D coordinates in
% 'Y' and and ground truth class labels in 'C'
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

[recalls, precisions] = compute_visualization_recalls_precisions(Y,C);
auc = compute_recalls_precisions_auc(recalls, precisions);
