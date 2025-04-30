function F = calculate_FIM(G, theta)

% F = calculate_FIM(G, theta)
% 
% This function calculates the Fisher information matrix given
% the G matrix whose elements are: 
% G(i, j) = p(y = i | x = j).
% and the r.v. X is assumed to have a categorical distribution with 
% probability vector theta

K = length(theta);
d = G*theta;
H = G(:, 1:K-1) - G(:, K);
F = H'*diag(1./d)*H;