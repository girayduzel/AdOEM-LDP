function G = make_G(K, k0, eps1, eps2)

% G = make_G(K, k0, eps1, eps2)
% 
% This function outputs the G matrix whose elements are:
% G(i, j) = p(y = i | x = j).
% K: number of categories
% k0: cardinality of the S
% eps1: first DP parameter
% eps2: second DP parameter

exp_eps1 = exp(eps1);
exp_eps2 = exp(eps2);

p11 = exp_eps1/(exp_eps1 + min(k0, K-1));
p12 = 1/(exp_eps1 + min(k0, K-1));
p22 = exp_eps2/(exp_eps2 + K - k0 - 1);
p21 = 1/(exp_eps2 + K - k0 - 1);

G = zeros(K);
G(1:k0, 1:k0) = (p11 - p12)*eye(k0) + p12*ones(k0);
G(k0+1:K, k0+1:K) = p11*((p22 - p21)*eye(K-k0) + p21*ones(K-k0));
G(k0+1:K, 1:k0) = p12/(K-k0);
G(1:k0, (k0+1):K) = p12;
