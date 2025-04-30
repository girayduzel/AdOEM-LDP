function [Theta_est, theta_est_final, Y, k_selected] = AdOEM_RR_LDP(X, eps_DP, eps1_coeff, theta0, alpha_EM, M_EM, c, loss_type)


T = length(X);
K = length(theta0);
Y = zeros(1, T);

% determine eps1 and eps2
if loss_type == 0
    eps1 = eps_DP;
    eps2_vec = zeros(1, K);
else
    eps1 = eps1_coeff*eps_DP;
    Sc_min = ceil(exp(eps_DP - eps1));
    k_vec = Sc_min:K;
    eps2_UB = eps_DP*ones(1, K);
    eps2_UB(k_vec) = (k_vec - 1)./(exp(eps1-eps_DP).*k_vec - 1);
    eps2_vec = min(eps_DP, eps2_UB);
end

GG = cell(1, K);
for k = 1:K
    GG{k} = make_G(K, k, eps1, eps2_vec(k));
end


theta_tilde = ones(K, 1)/K;
ord_ind_inv = zeros(1, K);
Theta_est = zeros(K, T);
k_selected = zeros(1, T);

P_yx = zeros(K, T);

theta_est = theta0;
for t = 1:T

    % Get x
    x = X(t);

    % A: Construct star set
    [theta_ord, ord_ind] = sort(theta_tilde, 'descend');
    ord_ind_inv(ord_ind) = 1:K;

    if loss_type == 0 % non-adaptive version
        k_best = K; % for precision reasons
    else
        L = zeros(1, K);
        for k = 1:K
            L(k) = calculate_utility(theta_ord, GG{k}, loss_type);
        end
        [~, k_best] = max(L);
    end
    star_set = ord_ind(1:k_best);
    k_selected(t) = k_best;

    % B: Sample y
    x_ord = ord_ind_inv(x);
    y_ord = find(rand <= cumsum(GG{k_best}(:, x_ord)), 1);
    Y(t) = ord_ind(y_ord);

    % C. Update theta
    % The likelihood vector
    p_yx = make_p_yx_vec(Y(t), K, star_set, k_best, eps1, eps2_vec(k_best));
    P_yx(:, t) = p_yx;

    pi_post_unnorm = log(theta_est) + log(p_yx);
    pi_post =  exp(pi_post_unnorm - log_sum_exp(pi_post_unnorm));

    % Online EM Update for theta
    gamma_t = t^(-alpha_EM);
    theta_est = (1-gamma_t)* theta_est + gamma_t * pi_post;

    % sample theta to be used in the generation of the next observation
    theta_tilde = gamrnd(c*t*theta_est, 1);
    theta_tilde = theta_tilde'/sum(theta_tilde);

    Theta_est(:, t) = theta_est;
end

% Make a few iterations of offline EM
theta_est_final = theta_est';
log_P_yx = log(P_yx);
for m = 1:M_EM
    log_Pi_post = log_P_yx + log(theta_est_final);
    m_pi = max(log_Pi_post, [], 1);
    log_sum_Pi_post = log(sum(exp(log_Pi_post - m_pi))) + m_pi;
    Pi_post = exp(log_Pi_post - log_sum_Pi_post);
    theta_est_final = mean(Pi_post, 2);
end