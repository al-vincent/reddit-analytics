function P = preprocess_input_similarities(P_in,method)
% preprocess input similarities
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

P = sparse(P_in);
% P = P_in;
P(1:size(P,1)+1:end) = 0;
switch method
    case {'ee', 'linlog', 'mdsks'}
        if nnz(P-P')>0
            P = 0.5 * (P+P');
        end
    case {'sne', 'nerv'}
        P = bsxfun(@rdivide, P, sum(P,2)+eps);
    case {'ssne', 'tsne', 'wtsne', 'gammane'}
        P = 0.5 * (P+P');
        P = P / sum(sum(P));
    otherwise
        error('unknown method');
end
