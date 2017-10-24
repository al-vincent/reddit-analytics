function dataclasses = clabel2dataclasses(clabel, nc, bSparse)
if ~exist('bSparse', 'var') || isempty(bSparse)
    bSparse = false;
end
n = length(clabel);
dataclasses = sparse((1:n)', clabel(:), ones(n,1), n, nc);
if ~bSparse
    dataclasses = full(dataclasses);
end
