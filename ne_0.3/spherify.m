function Y = spherify(Y_in)
% Find the closest sphere of given the input Y coordinates
Y = Y_in;
Y = bsxfun(@minus, Y, mean(Y));
Ynorm = sqrt(sum(Y.^2,2));
radius = mean(Ynorm);
Y = bsxfun(@rdivide, Y, Ynorm) * radius;
