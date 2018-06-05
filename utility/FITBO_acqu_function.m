function Mutual_info = FITBO_acqu_function(x, x_es, y_es,KernelMatrixInv, l, sigma,sigma0, eta, N_hypsample)
 
 % Function: compute the mutual information ofthe FITBO acquisition function
 d          = size(x_es,2);             % dimension of input
 n          = size(x, 1);               % number of test point
 Meanl      = zeros(N_hypsample, n);    
 Varl       = zeros(N_hypsample, n);
 Entropy_20 = zeros(N_hypsample, n);

 e          = exp(1);                   % exponential term
 
for i = 1 : N_hypsample


        % compute the observation measurement for g(x):g=sqrt(2*(y-eta))
        g_es    = sqrt(2 .*(y_es - eta(i)));

        % compute posterior mean and variance for g(x)
        [mg,var] = mean_var(x, x_es, g_es, KernelMatrixInv{i}, l(i,:), sigma(i), sigma0(i));
        % approximated posterior mean and variance of y(x) using linearisation
        meanl   = 0.5 * mg .^ 2 + eta(i);
        varl   = mg.^2 .* var + sigma0(i) ;  
        
        % store the posterior mean and variance of y(x)
        Meanl(i,:) = meanl;
        Varl(i,:)  = varl ;

        % compute the entropy of a single Gaussian at a particular hyperparameter sample value
        Entropy_20(i,:) = 0.5 .* log(2 .* pi .* e .* varl);

end

        % compute the 2nd entropy term by marginalising over all hyperparameter values 
        Entropy_2   = mean (Entropy_20);
    
        %%%%%%%%%% COMPUTE THE 1ST ENTROPY TERMcompute the 1st entropy term %%%%%%%%%%
        % compute the weights for gaussian mixtures
        weights = ones(N_hypsample,1)./N_hypsample;
        % create an empty array for data storage
        Entropy_1   = zeros(1, n);
        
        % Method='Integration by quadrature';
        for j2 = 1 : n
            Entropy_1(j2)    = computeEntropyGMM1Dquad(Meanl(:, j2), Varl(:, j2), weights, 3);
            
        end

        %%%%%%%%%% COMPUTE MUTUAL INFORMATION AND FIND THE NEXT EVALUATION %%%%%%%%%%
        % compute the true mutual information
        Mutual_info_original = Entropy_1 - Entropy_2;    
        % compute the negative of the mutual information for minimisation
        Mutual_info          = - Mutual_info_original;
end
