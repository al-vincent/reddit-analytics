function weights = get_weights(P)
% wrapper, to calculate the importance of data points from P
% currently by degree centrality
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

weights = sum(P)';
