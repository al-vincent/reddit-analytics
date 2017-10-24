function [obj,grad] = compute_nerv_obj_grad_exact(Y,P,Pnz,I,J,lambda,epsilon)
n = size(Y,1);

indnz = sub2ind(size(P), I, J);
logepsilon = log(epsilon);
logPnz = log(Pnz+eps);

q = exp(-distSqrdSelf(Y));
q(1:n+1:end) = 0;
Q = bsxfun(@rdivide, q, sum(q,2));
Qnz = Q(indnz);
logQnz = log(Qnz+eps);

recall = -sum((Pnz-epsilon).*logQnz) ...
         - epsilon*sum(sum(log(Q+eye(n)))) ...
         + sum(Pnz.*logPnz) + epsilon*logepsilon*(n*n-nnz(P));
precision = sum(Qnz.*(logepsilon+logQnz-logPnz)) - logepsilon*sum(sum(Q));
obj = lambda * recall + (1-lambda) * precision;

if nargout>1
    nnzP = nnz(P);
    B = double(P>0);
    gradr = GraphLaplacian((P+P')-(Q+Q')-epsilon*(B+B')) * Y;
    Qtmp = Qnz.*(logPnz - logQnz - 1 - logepsilon);
    Mp = sparse(I,J,Qtmp,n,n,nnzP) ...                                % sparse part
        + bsxfun(@times, Q, -full(sum(sparse(I,J,Qtmp,n,n,nnzP),2))); % dense part
    gradp = GraphLaplacian(Mp+Mp') * Y;

    grad = 2*(lambda * gradr + (1-lambda) * gradp);
    grad = grad(:);
end
