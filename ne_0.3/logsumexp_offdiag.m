function s = logsumexp_offdiag(a, dim)
% Returns log(sum(exp(a),dim)) while avoiding numerical underflow.
% Default is dim = 1 (columns).
% logsumexp(a, 2) will sum across rows instead of columns.
% Unlike matlab's "sum", it will not switch the summing direction
% if you provide a row vector.

% Written by Tom Minka
% (c) Microsoft Corporation. All rights reserved.
%
% Modified by Zhirong Yang, to omit diagonal elements of a

if nargin < 2
  dim = 1;
end

n = size(a,1);
a_offdiag = a;
a_offdiag(1:n+1:end) = -inf;

% subtract the largest in each column
[y, i] = max(a_offdiag,[],dim);
dims = ones(1,ndims(a_offdiag));
dims(dim) = size(a_offdiag,dim);
a_offdiag = a_offdiag - repmat(y, dims);
s = y + log(sum(exp(a_offdiag),dim));
i = find(~isfinite(y));
if ~isempty(i)
  s(i) = y(i);
end
