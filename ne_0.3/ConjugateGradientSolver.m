function x = ConjugateGradientSolver(A,b,x0,pfun)
% Solve Ax=b by conjugate gradient solver
% If pfun is given (by e.g. pfun=cmg_sdd(A)), it calls pcg for faster
% convergence
% x0 can be set to the solution in the last iteration
%
% Copyright (c) 2014, Zhirong Yang (Aalto University)
% All rights reserved.


if ~exist('pfun', 'var') || isempty(pfun)
    x = cgLanczos( A, b, false, false, 1000, 0, x0 );
else
    tol = 1e-4;
    maxit = 100;
    [x,~] = pcg(A, b, tol, maxit, pfun, [], x0);
end
