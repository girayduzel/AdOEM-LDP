function L = calculate_utility(theta, G, util_type)

% L = calculate_utility(theta, G, util_type)
% 
% This function computes the utility function given the utility type.
% theta: current sample of probabilities for categorical distribution
% G: the stochastic matrix where G(i, j) = p(y = i | x = j)
% loss_type: a value between 1 and 6 to choose one of loss functions below

% theta must be ordered
theta = sort(theta, 'descend');

switch util_type
    case 1 % FIM
        F = calculate_FIM(G, theta);
        L = -trace(inv(F));
    case 2 % entropy
        p = G*theta;
        L = sum(log(p).*p);
    case 3 % TV between p(x) and p(y)
        L = -sum(abs(theta - G*theta));
    case 4 % TV between p(x) and p(x|y)
        py = G*theta;
        p_x_post_minus_px_mtx = (G.*theta')./py - theta';
        L = sum(sum(abs(p_x_post_minus_px_mtx), 2).*py);
    case 5 % Expected mean squared error
        L = -1 + sum((G.^2)*(theta.^2)./(G*theta));
    case 6 % Prob(Y = X)
        L = sum(theta.*diag(G));
end
