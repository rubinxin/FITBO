function logpo=logposter_pos(ln_hypsamples,x_es,y_es,mean_ln_yminob_minus_eta,var_ln_yminob_minus_eta)
% Function: compute log of the posterior distribution 
 
    % theta=[length scale d1; length scale d2;.... ;output scale;noise variance;yminob - eta]
    theta         = exp(ln_hypsamples);
    
    d             = size(x_es, 2);
    
    % determinant of log characteristic length matrix
    ldeterminant  = prod(theta(1:d));
    
    % log-likelihood of the data with the log of a particular hyperparameter sample
    loglikelihood = loglike(theta, x_es, y_es);
    
    % prior distribution of the log charateristic length
    L_prior_mu    = 0.3.*ones(d,1);
    L_prior_var   = 3.*ones(d,1);
    plength       = Gaussianprior(ln_hypsamples(1:d), log(L_prior_mu'), diag(L_prior_var')) ./ ldeterminant;
    
    % prior distribution of the log output scale
    plsig         = Gaussianprior(ln_hypsamples(d+1), log(1),3.0) ./ theta(d+1);
    
    % prior distribution of the log noise variance 
    plvarn0       = Gaussianprior(ln_hypsamples(d+2), log(1e-3),0.1) ./ theta(d+2);
    
    % prior distribution of log ( y_minob - eta )
    plymin_eta    = Gaussianprior(ln_hypsamples(d+3), mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta) ./ theta(d+3);

    % compute log posterior distribution 
    logpo         = loglikelihood + log( plength * plsig * plvarn0 * plymin_eta );
    
end
    
     