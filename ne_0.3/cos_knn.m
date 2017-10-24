function A = cos_knn(X, k)
N = size(X,1);
if ~issparse(X) && N>20000
    error('large and dense data matrix is not affordable!');
end

indz = full(sum(X,2))==0;
X = bsxfun(@rdivide, X, sqrt(sum(X.^2,2)));
X(indz,:) = 0;

C = X * X';
bCloseToDense = nnz(C)/numel(C)>0.5;
if bCloseToDense
    C = full(C);
end
clear X;
C(1:N+1:end) = 0;
[~,ind] = sort(C,'descend');
IX = ind(1:k,:);

if bCloseToDense
    indCz = C==0;
    clear C;
end
A = sparse(reshape(repmat(1:N, k, 1), N*k, 1), reshape(IX(1:k,:), N*k, 1), ones(N*k,1), N, N);
if bCloseToDense
    A(indCz) = 0;
    A = sparse(A);
else
    A = sparse(A .* (C>0));
end

