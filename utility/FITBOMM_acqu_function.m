function [Mutual_info, dMutual_info] = FITBOMM_acqu_function(x, x_es, y_es,KernelMatrixInv, l, sigma,sigma0, eta, N_hypsample)
 
 % Function: compute the mutual information ofthe FITBO acquisition function
 d          = size(x_es,2);             % dimension of input
 n          = size(x, 1);               % number of test point
 if nargout == 1    
     Meanl      = zeros(N_hypsample, n);    
     Varl       = zeros(N_hypsample, n);
     Entropy_20 = zeros(N_hypsample, n);

     e          = exp(1);                   % exponential term
     weights     = 1 ./ N_hypsample;

    for i = 1 : N_hypsample

            % compute the observation measurement for g(x)?g=sqrt(2*(y-eta))
            g_es    = sqrt(2 .*(y_es - eta(i)));

            % compute posterior mean and variance for g(x)
            [mg,var] = mean_var(x, x_es, g_es, KernelMatrixInv{i}, l(i,:), sigma(i), sigma0(i));
            % approximated posterior mean and variance of y(x) using linearisation
            meanl   = 0.5 * mg .^ 2 + eta(i);
            varl   = mg.^2 .* var + sigma0(i) ;  
            % store the posterior mean and variance of y(x)
            Meanl(i,:) = meanl;
            Varl(i,:)  = varl ;

            % compute the entropy of a Gaussian at a particular hyperparameter sample value
            Entropy_20(i,:) = 0.5 .* log(2 .* pi .* e .* varl);

  
    end

            % compute the 2nd entropy term by marginalising over all hyperparameter values 
            Entropy_2   = mean (Entropy_20);

            %%%%%%%%%% COMPUTE THE 1ST ENTROPY TERMcompute the 1st entropy term %%%%%%%%%%
            % compute the weights for gaussian mixtures

            mean_GMM    = weights .* sum( Meanl ); 
            % compute covariance of GMM distribution
            cov_GMM     = weights .* sum( Varl + Meanl .^2 ) - mean_GMM .^2; 

            % create an empty array for data storage
            Entropy_1   = 0.5 .* log( 2 .* pi .* e .* cov_GMM );

            %%%%%%%%%% COMPUTE MUTUAL INFORMATION AND FIND THE NEXT EVALUATION %%%%%%%%%%
            % compute the true mutual information
            Mutual_info_original = Entropy_1 - Entropy_2;    
            % compute the negative of the mutual information for minimisation
            Mutual_info          = Mutual_info_original;
 else
     Meanl      = zeros(N_hypsample, n);    
     Varl       = zeros(N_hypsample, n);
     Entropy_20 = zeros(N_hypsample, n);
     dMmeanl      = zeros(N_hypsample, d);    
     dEntropy_20 = zeros(N_hypsample, d);
     dEntropy_11 = zeros(N_hypsample, d);

     e          = exp(1);                   % exponential term
     weights     = 1 ./ N_hypsample;

    for i = 1 : N_hypsample

            % compute the observation measurement for g(x)?g=sqrt(2*(y-eta))
            g_es    = sqrt(2 .*(y_es - eta(i)));

            % compute posterior mean and variance for g(x)
            [mg,var,dmg,dvar] = mean_var(x, x_es, g_es, KernelMatrixInv{i}, l(i,:), sigma(i), sigma0(i));
            % approximated posterior mean and variance of y(x) using linearisation
            meanl   = 0.5 * mg .^ 2 + eta(i);
            varl   = mg.^2 .* var + sigma0(i) ;  
            % store the posterior mean and variance of y(x)
            Meanl(i,:) = meanl;
            Varl(i,:)  = varl ;

            % compute the entropy of GP of y(x) at a particular hyperparameter sample value
            Entropy_20(i,:) = 0.5 .* log(2 .* pi .* e .* varl);
            
            mg2   = repmat(mg,[d,1]);
            var2  = repmat(var,[d,1]);
            meanl2  = repmat(meanl,[d,1]);
            varl2   = repmat(varl,[d,1]);
            
            % compute gradient
            dmeanl = mg2 .* dmg;                               
            dvarl  = 2 .* mg2 .* var2 .* dmg + mg2.^2 .* dvar;
            dEntropy_11(i,:) = dvarl + 2 .* meanl2 .* dmeanl;
            dMmeanl(i,:)     = dmeanl;
            dEntropy_20(i,:) = dvarl ./(2 .* varl2);
            
    end
            % compute the 2nd entropy term by marginalising over all hyperparameter values 
            Entropy_2   = mean (Entropy_20);
            
            %%%%%%%%%% COMPUTE THE 1ST ENTROPY TERMcompute the 1st entropy term %%%%%%%%%%
            % compute the weights for gaussian mixtures
            mean_GMM    = weights .* sum( Meanl ); 
            % compute covariance of GMM distribution
            cov_GMM     = weights .* sum( Varl + Meanl .^2 ) - mean_GMM .^2; 
            Entropy_1   = 0.5 .* log( 2 .* pi .* e .* cov_GMM );
            
            %%%%%%%%%% COMPUTE MUTUAL INFORMATION AND FIND THE NEXT EVALUATION %%%%%%%%%%
            % compute the true mutual information
            Mutual_info_original = Entropy_1 - Entropy_2;    
            Mutual_info = Mutual_info_original;
            
            %%%%%%%%%% compute the gradient of mutual information %%%%%%%
            dEntropy_1  = ( mean( dEntropy_11 ) - 2 .* mean_GMM .* mean(dMmeanl))./ (2.* cov_GMM);
            dEntropy_2  = mean( dEntropy_20 );            
            dMutual_info = (dEntropy_1 - dEntropy_2)';
           
 end
