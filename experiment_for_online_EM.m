% The main file to conduct the numerical experiments
clear; clc; close all;

rng(1); % Setting the seed to obtain reproducible results

% DP parameters
eps_DP_vec = [0.5 1 2]; L_e = length(eps_DP_vec);

% Loss types
loss_type_vec = [0 6]; L_m = length(loss_type_vec);

K_vec = [10, 20, 50]; L_K = length(K_vec); % Number of categories

% Dirichlet parameters
rho_coeff_vec = [0.01 0.1 1]; L_r = length(rho_coeff_vec); 
MC_run = 50; % Number of Monte Carlo runs

% Initialization of cells for TV results 
% and cardinalities of the selected subsets
TV1_RR = repmat({zeros(L_e, L_m, MC_run)}, L_K, L_r);
TV2_RR = repmat({zeros(L_e, L_m, MC_run)}, L_K, L_r);

K_selected = cell(L_K, L_r, L_e, L_m, MC_run);

% Coefficient that relates eps1 to eps_DP
% eps1 = eps1_coeff*eps_DP

eps1_coeff = 0.9;

% Name of the file for saving the results
datanametosave = ['online_EM_', 'K_vec_' sprintf('%d_', K_vec) sprintf('eps1coeff_%02d_', 100*eps1_coeff) ...
    'methods' sprintf('_%d', loss_type_vec)]; 

for mc = 1:MC_run
    for i1 = 1:L_K
        
        K = K_vec(i1);
        T = 1000*K; 
        alpha_EM = 0.65;
        M_EM = 1000;
        theta0 = ones(1, K)/K;
        c = 0.01;
        for i2 = 1:L_r
            rho_coeff = rho_coeff_vec(i2);

            rho_x = rho_coeff*ones(1, K);
            theta_true = gamrnd(rho_x, 1);
            theta_true = theta_true/sum(theta_true);
            X = randsample(1:K, T, 'true', theta_true);

            for i3 = 1:L_e
                eps_DP = eps_DP_vec(i3);
                for i4 = 1:L_m
                    loss_type = loss_type_vec(i4);
                    fprintf('MC run no %d for RR, \n epsilon = %.2f, rho = %.2f, K=%d, method=%d... \n', mc, eps_DP, rho_coeff, K, loss_type);
                    
                    [Theta_est, theta_est_final, Y, k_selected] = AdOEM_RR_LDP(X, eps_DP, eps1_coeff, theta0, alpha_EM, M_EM, c, loss_type);

                    TV_current1 = 0.5*sum(abs(Theta_est(:, end)' - theta_true));
                    TV_current2 = 0.5*sum(abs(theta_est_final' - theta_true));
                    TV1_RR{i1, i2}(i3, i4, mc) = TV_current1;
                    TV2_RR{i1, i2}(i3, i4, mc) = TV_current2;
                    K_selected{i1, i2}(i3, i4, mc) = mean(k_selected);
                    disp([TV_current1 TV_current2]);
                    end
                end
            end
            save(datanametosave);
        end
    end

%% results
% Computing the means and standard deviations of the TV results
TV1_RR_mean = cell(L_K, L_r); 
TV1_RR_stdev = cell(L_K, L_r);
TV2_RR_mean = cell(L_K, L_r); 
TV2_RR_stdev = cell(L_K, L_r);
for i1 = 1:L_K
    for i2 = 1:L_r
        TV1_RR_mean{i1, i2} = mean(TV1_RR{i1, i2}, 3);
        TV1_RR_stdev{i1, i2} = std(TV1_RR{i1, i2}, [], 3);
        TV2_RR_mean{i1, i2} = mean(TV2_RR{i1, i2}, 3);
        TV2_RR_stdev{i1, i2} = std(TV2_RR{i1, i2}, [], 3);
    end
end

save(datanametosave);