function lntheta_eta = samplehyperparameters(x_es,y_es, Nsamples,mean_ln_yminob_minus_eta,var_ln_yminob_minus_eta, ess)
    
    d=size(x_es,2);
    burnin = 300;
    
    % define the log distri from which we sample 
    % log_like_fn  = @(x)loglike(x, x_es, y_es,mean_ln_yminob_minus_eta);
    log_like_fn = @(x)logposter_pos(x,x_es,y_es,mean_ln_yminob_minus_eta,var_ln_yminob_minus_eta);
    
    % initial guess
    ln_theta0 = log([0.3 .* ones(d,1); 1.0; 1.0e-2; exp(mean_ln_yminob_minus_eta) ]);
    
    % use elliptical slice sampler
    if ess == 1 
                
        xx        =  ln_theta0'; 
        % specify the prior covariance of log of hyperparameter samples
        a =1;
        cholsigma  = chol(diag([ a * ones(d,1); a; a; a * var_ln_yminob_minus_eta]));
        
        % create matrix for storing samples
        samples     = zeros(Nsamples+burnin, length(xx));

        for i=1 : (Nsamples+burnin)
            % draw a sample using elliptical slice sampler
            xx  = gppu_elliptical(xx, cholsigma, log_like_fn, []);
            samples(i,:)       = xx;
        end
        % output samples after burnin
        lntheta_eta = samples(burnin+1:end,:);
    
    % use normal slice sampler    
    else 
        lntheta_eta = slicesample(ln_theta0,Nsamples,'logpdf',log_like_fn, 'thin',2,'burnin',burnin);
    end
end
    
    


 