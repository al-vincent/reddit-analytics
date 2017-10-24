function [recalls, precisions] = compute_visualization_recalls_precisions(Y,C)
% compute the recalls and precisions of sampled points for given 2-D 
% coordinates in 'Y' and and ground truth class labels in 'C'
%
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

n = size(Y,1);
nSample = min(n,5000);

recalls = zeros(n-1,1);
precisions = zeros(n-1,1);

retrieved = (1:n-1)';

% rand('seed', 0);
rng('default');
selected_ind = randi(n, nSample, 1);

for t=1:nSample
    if mod(t,1000)==0
        fprintf('t=%d\n', t);
    end
    i = selected_ind(t);
    dist2 = distSqrd(Y(i,:),Y);
    dist2(i) = -1; % to ensure correct ordering
    [~, ind] = sort(dist2);
    bc = C(ind(2:end))==C(i);
    if sum(bc)==0 % skip the samples in the singleton classes
        continue;
    end
    
    hits = cumsum(bc);
    precisions = precisions + hits ./ retrieved;
    recalls = recalls + hits / sum(bc);
end

precisions = precisions / nSample;
recalls = recalls / nSample;
