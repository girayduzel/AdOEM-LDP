function g = make_p_yx_vec(y, K, S, k0, eps1, eps2)

% g = make_p_yx_vec(y, K, S, k0, eps1, eps2)
% 
% This function creates the likelihood vectors, P(y|x).
% y: privatized observations
% K: number of categories
% S: subset of indices
% k0: cardinality of S
% eps1: first DP parameter
% eps2: second DP parameter

exp_eps_1 = exp(eps1);
exp_eps_2 = exp(eps2);

p11 = exp_eps_1/(exp_eps_1 + min(k0, K-1));
p12 = 1/(exp_eps_1 + min(k0, K-1));
p21 = 1/(exp_eps_2 + K - k0 - 1);
p22 = exp_eps_2/(exp_eps_2 + K - k0 - 1);

cond = sum(y == S) > 0;
if cond == 0
    g = p11*p21*ones(1, K);
    g(S) = p12*(1/(K - k0));
    g(y) = p11*p22;
else
    g = p12*ones(1, K);
    g(y) = p11;
end
