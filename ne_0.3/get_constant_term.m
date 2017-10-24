function constant = get_constant_term(P,weights,Pnz,I,J,method, varargin)
% Calculate the constant term of the specific visualizaiton method
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

switch method
    case 'sne'
        constant = sum(Pnz.*log(Pnz+eps));
    case 'ssne'
        constant = sum(Pnz.*log(Pnz+eps));
    case {'tsne', 'tsne_3d'}
        constant = sum(Pnz.*log(Pnz+eps));
    case {'wtsne', 'wtsne_3d'}
        constant = sum(Pnz.*log(Pnz+eps))-2*sum(P*log(weights));
    case {'nerv', 'ee', 'linlog'}
        constant = 0;
    case 'mdsks'
        constant = 0.5 * log(Pnz'*Pnz);
    case 'gammane'
        n = size(P,1);
        gamma = varargin{1};
        switch gamma
            case 0
                constant = sum(Pnz.*log(Pnz+eps));
            case -1
                constant = -log(n*(n-1))-sum(log(Pnz+eps))/(n*(n-1));
            otherwise
                constant = log(sum(Pnz.^(gamma+1)))/(gamma*(gamma+1));
        end
    otherwise
        error('unknown method');
end
