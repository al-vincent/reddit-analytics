function [ x, istop, itn, Anorm, Acond, rnorm, xnorm, D ] = ...
           cgLanczos( A, b, show, check, itnlim, rtol, x0 )

%        [ x, istop, itn, Anorm, Acond, rnorm, xnorm, D ] = ...
%          cgLanczos( A, b, show, check, itnlim, rtol );
%
% cgLanczos solves the system of linear equations Ax = b,
% where A is an n x n positive-definite symmetric matrix
% and b is a given n-vector, where n = length(b).
%  
% "A" may be a dense or sparse matrix (preferably sparse!)
% or a function handle such that y = A(x) returns the product
% y = A*x for any given vector x.
%
% On entry:
% show         (true/false) controls the iteration log.
% check        (true/false) controls the test for symmetry of A.
% itnlim       (integer) limits the number of iterations.
% rtol         (e.g. 1e-8) is the requested accuracy.  Iterations
%              terminate if rnorm < (Anorm*xnorm)*rtol.
%
% On exit:
% x            (n-vector) is the solution estimate
% istop        (0--6) gives reason for termination (see "msg" below)
% itn          (integer) is the number of CG iterations
% Anorm        estimates the Frobenius norm of A
% Acond        estimates the condition of A (in F-norm)
% rnorm        estimates the residual norm:  norm(r) = norm(b-Ax)
% xnorm        is the exact norm(x)
% D            (n-vector) estimates diag(inv(A)).


% NOTE: If a preconditioner is known, say M = C'*C ~= A,
% we need to work with quantities defined by
%    Abar = inv(C')*A*inv(C)
%    bbar = inv(C')*b
%    Abar*xbar = bbar
%    x    = inv(C)*xbar.
% This means:
%    Implement a function y = Abar(x) that forms
%       y = C\x;
%       y = A*y;
%       y = (C')\y;
%    Solve the transformed problem via
%       [xbar, istop, ..., D] = cgLanczos( Abar, bbar, ... );
%    Recover the solution via
%       x = C\xbar;
%
% Beware that D then estimates diag(inv(Abar)), which will
% be close to the identity if M is a good preconditioner.


% Code author: Michael Saunders, SOL, Stanford University.
%
% Reference for use of Lanczos process for solving equations:   
%              C. C. Paige and M. A. Saunders (1975),
%              Solution of sparse indefinite systems of linear equations,
%              SIAM J. Numer. Anal. 12(4), pp. 617-629.
%
% 22 Oct 2007: First version of cgLanczos.m.
%              We haven't had a Lanczos-based CG routine before.
%              The aim is to return D ~= diag(inv(A)), as requested by
%              Giannis Chantas <chanjohn@cs.uoi.gr>
%              Dept of Computer Science, University of Ioannina, Greece.
% 
% 10 Aug 2010: Lanczos p = Av;  alpha = v'*p;  p = p - alpha*v - beta*v1;
%              changed to slightly better
%              p - Av;  p = p - beta*v1;  alpha = v'*p;  p = p - alpha*v;
%---------------------------------------------------------------------

debug  = false;
n      = length(b);
if n>100, debug = false; end   % Safeguard!

if show
   fprintf('\n')
   fprintf('\n Enter cgLanczos.   Solution of symmetric Ax = b')
   fprintf('\n n = %6g    itnlim = %6g    rtol = %11.2e', n,itnlim,rtol)
end

if ~exist('x0', 'var') || isempty(x0)
    x0 = zeros(n,1);
end

istop = 0;   itn   = 0;
Anorm = 0;   Acond = 0;   x = x0;
rnorm = 0;   xnorm = 0;   D = zeros(n,1);


%---------------------------------------------------------------------
% Decode Aname.
%---------------------------------------------------------------------
if isa(A,'double')         % A is an explicit matrix.
    if show
        if issparse(A)
            fprintf('\n A is an explicit sparse matrix')
            fprintf('\n\n nnz(A) =%9g', nnz(A))
        else
            fprintf('\n\n A is an explicit dense matrix' )
        end
    end
elseif isa(A,'function_handle')
  fprintf('\n\n A is defined by function handle %s', func2str(A))
else
  error('cgLanczos','A must be a matrix or a function handle')
end

if check                    % See if A is symmetric.
  w1   = (1:n)';
  w2   = 1./w1;
  s    = w2'*LanczosxxxA( A,w1 );
  t    = w1'*LanczosxxxA( A,w2 );
  z    = abs(s-t);
  epsa = (s+t+eps)*eps^0.333333;
  if z > epsa
    istop = 5;  done = true;  show = true;   % A seems to be unsymmetric
  end
  clear w1 w2
end

%------------------------------------------------------------------
% Set up the first Lanczos vector v.
%------------------------------------------------------------------
done  = false;
beta1 = norm(b);
if beta1==0
  istop = 0;  done = true;  show = true;     % b=0 exactly.  Stop with x = 0.
else
  v   = (1/beta1)*b;
end

oldb   = 0;   beta   = beta1;   rnorm  = beta1;
Tnorm2 = 0;   Wnorm2 = 0;

if show
  fprintf('\n\n   Itn     x(1)       norm(r)    norm(x)  norm(A)  cond(A)')
  fprintf('\n %6g %12.5e %10.3e', itn, x(1), beta1)
end

%---------------------------------------------------------------------
% Main iteration loop.
% --------------------------------------------------------------------
if ~done                        % k = itn = 1 first time through
  while itn < itnlim
    itn = itn + 1;

    %-----------------------------------------------------------------
    % Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
    % The general iteration is similar to the case k = 2.
    %   p      = A*v2
    %   alpha2 = v2'*p
    %   p      = p - alpha2*v2 - beta2*v1
    %   beta3  = norm(p)
    %   v3     = (1/beta3)*p.
    %-----------------------------------------------------------------
    p     = LanczosxxxA( A,v );
    if itn>1
      p   = p - beta*v1;
    end
    alpha = v'*p;                        % alpha = v'Av in theory
    if alpha<=0, istop = 6;  break; end  % A is indefinite or singular
    p     = p - alpha*v;

    oldb  = beta;               % oldb = betak
    beta  = norm(p);            % beta = betak+1
    beta  = max(beta,eps);      % Prevent divide by zero
    v1    = v;
    v     = (1/beta)*p;

    if itn==1                   % Initialize a few things.
      delta  = sqrt(alpha);     % delta1 = sqrt(alpha1)
      gamma  = beta /delta;     % gamma2 = beta2/delta1
      zeta   = beta1/delta;     % zeta1  = beta1/delta1
      w      = v1   /delta;     % w1     = v1   /delta1
      x      = zeta*w;          % x1     = w1*zeta1  = v1*(beta1/alpha1)
      D      = w.^2;            % Initialize diagonals of Wk Wk'
      Tnorm2 = alpha^2 + beta^2;
      if debug
        Tk = alpha;
        Vk = v1;
        Wk = w;
        b1 = beta1;
      end

    else                        % Normal case (itn>1)
      delta  = alpha - gamma^2;
      if delta<=0, istop = 6;  break; end  % Tk is indefinite or singular
      delta  = sqrt(delta);
      zeta   =     - gamma*zeta/delta;
      w      = (v1 - gamma*w  )/delta;
      x      = x + zeta*w;
      D      = D + w.^2;
      gamma  = beta /delta;
      Tnorm2 = Tnorm2 + alpha^2 + oldb^2 + beta^2;
      if debug
	tk = [zeros(itn-2,1)
                  oldb      ];
        Tk = [Tk   tk
              tk'  alpha];
        Vk = [Vk v1];
        Wk = [Wk w ];
	b1 = [    beta1
	      zeros(itn-1,1)];
      end
    end

    if debug
      yk = Tk\b1;
      xk = Vk*yk;
      disp(' ')
      xdiff = norm(x-xk)
      keyboard
    end

    %-----------------------------------------------------------------
    % Estimate various norms and test for convergence.
    %-----------------------------------------------------------------
    Wnorm2 = Wnorm2 + norm(w)^2;
    Anorm  = sqrt( Tnorm2 );
    Acond  = Anorm * sqrt(Wnorm2);
    xnorm  = norm(x);
    epsa   = Anorm * eps;
    rnorm  = abs(beta*zeta/delta);
    test1  = rnorm / (Anorm*xnorm);    %  ||r|| / (||A|| ||x||)

    % See if any of the stopping criteria are satisfied.

    if itn   >= itnlim , istop = 4; end
    if Acond >= 0.1/eps, istop = 3; end
    if test1 <= eps    , istop = 2; end
    if test1 <= rtol   , istop = 1; end

    % See if it is time to print something.

    if show
      prnt   = false;
      if n      <= 40       , prnt = true; end
      if itn    <= 10       , prnt = true; end
      if itn    >= itnlim-10, prnt = true; end
      if mod(itn,10)==0     , prnt = true; end
      if Acond  >= 1e-2/eps , prnt = true; end
      if test1  <= 10*eps   , prnt = true; end
      if test1  <= 10*rtol  , prnt = true; end
      if istop  ~=  0       , prnt = true; end

      if prnt
	fprintf('\n %6g %12.5e %10.3e %8.1e %8.1e %8.1e', ...
	        itn,x(1),rnorm,xnorm,Anorm,Acond);
      end
      if mod(itn,10)==0, fprintf('\n'); end
    end % show

    if istop > 0, break; end
  end % main loop
end % if ~done early

%-------------------------------------------------------------------
% Display final status.
%-------------------------------------------------------------------
msg = ['beta1 = 0.  The exact solution is  x = 0    '   % istop = 0
       'A solution to Ax = b was found, given rtol  '   %         1
       'Maximum accuracy achieved, given eps        '   %         2
       'Acond has exceeded 0.1/eps                  '   %         3
       'The iteration limit was reached             '   %         4
       'A does not define a symmetric matrix        '   %         5
       'A does not define a positive-definite matrix']; %         6
msg = msg(istop+1,:);

if show
  fprintf('\n')
  fprintf('\n %s', msg)
  fprintf('\n istop =%3g             itn   =%10g', istop,itn)
  fprintf('\n Anorm =%10.2e      Acond =%10.2e', Anorm,Acond)
  fprintf('\n rnorm =%10.2e      xnorm =%10.2e', rnorm,xnorm)
  fprintf('\n')
end
%-----------------------------------------------------------------------
% End function cgLanczos
%-----------------------------------------------------------------------


function y = LanczosxxxA( A,x )

% gives  y = Ax for a matrix A defined by parameter A.

  if isa(A,'function_handle')
    y = A(x);
  else
    y = A*x;
  end
%-----------------------------------------------------------------------
% End private function LanczosxxxA
%-----------------------------------------------------------------------
