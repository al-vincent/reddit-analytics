function [Y, Cs, ts, Ys, extra] = optimizer_mm_backtrack_order1(P, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, method, recordYs, exact, varargin)
% minimization by Majorization-Minimization, first order surrogate
% 
% Copyright (c) 2016, Zhirong Yang
% All rights reserved.

order = 1;
[Y, Cs, ts, Ys, extra] = optimizer_mm_backtrack_various_order(P, Y0, attr, sphere, theta, max_iter, check_step, tol, max_time, verbose, method, recordYs, exact, order, varargin{:});
