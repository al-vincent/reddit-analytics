function LP = get_LP(P,method)
% Calculate the Laplacian (approximated Hessian) of the specific visualizaiton method
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

switch method
    case 'sne'
        LP = GraphLaplacian((P+P')/2);
    case {'ssne', 'ee', 'linlog'}
        LP = GraphLaplacian(P);
    case {'tsne', 'wtsne', 'tsne_3d', 'wtsen_3d'}
        LP = GraphLaplacian(P);
    case 'nerv'
        LP = GraphLaplacian((P+P')/2);
    case 'mdsks'
        LP = GraphLaplacian(P);
    case 'gammane'
        LP = GraphLaplacian(P);
    otherwise
        error('unknown method');
end
LP = LP + speye(size(P,1))*1e-10*min(sum((P+P')/2));
