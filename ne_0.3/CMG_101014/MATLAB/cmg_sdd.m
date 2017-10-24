% function pfun = cmg_sdd(A,opts)
%
% A     : SDD matrix with non-positive off-diagonals
%
% pfun : function implementing a preconditioner for A 
%
%
% For an arbitrary b-side in the null space of A, 
%    the system Ax = b can be solved by 
%    x = pcg(A, b, tol,iter, pfun);        % >> help pcg for documentation


% update: directed forest indicating previously computed first level clustering
% update:  directed forest indicating new first level clustering


% stats : hierarchy statistics, a cell of structs, one struct per level
%       : each cell struct contains self-explanatory info



function [pfun H] = cmg_sdd(A,opts) %#ok<STOUT,FNDEF>


if ~isfield(opts, 'display')
    opts.display = 1;
end

n = length(A);
dA = (sum(A))';
%dA(dA<0) = 0;
dA = (dA+abs(dA))/2;
s = dA./diag(A);
not_dominant = find(s<1e-14);

A = abs(A-diag(diag(A)));

if (length(not_dominant)<n) 
    lap =0;
    opts.d = dA;
    [pfun H] = cmg_dd(A,opts);
else
    lap =1;
    [pfun H] = cmg_dd(A,opts);
end





