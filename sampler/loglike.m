
function loglikelihood = loglike(theta_eta, x_es, y_es)
 % Function for negatie marginal likelihood     
 % Input: x_es=Nxd ; y_es=Nx1 ; 
 % ln_hypsamples=log([length scale d1; length scale d2;.... ;output scale;noise variance; eta])
 % Output: loglikelihood = 1x1

 % find the minimum value observed
    yminob  = min(y_es);
    
 % dimension of the input
    d       = size(x_es, 2);
    l = 1./theta_eta(1:d).^2; 
    sigma = theta_eta(d+1);
    sigma0 = theta_eta(d+2);
    eta     = yminob - theta_eta(d+3);

 % get the value of hyperparameters including (yminob - eta)
%     theta_eta = exp(ln_hypsamples);
    
 % extract the hyperparameters except eta so theta is a vector contain all hyperparameters: theta_eta=[sig,length,varn,eta]
%     theta   = theta_eta(1 : d+2);
    
 % compute eta value
 % compute the observation measurement for g(x)?g=sqrt(2*(y-eta))  
    g_es    = sqrt(2 .* (y_es - eta));
    
 % computes the noisy kernel between observation data points using RBF method

    Kn      = computeKmm(x_es, l, sigma, sigma0);
    K_testob = computeKnm(x_es, x_es, l, sigma);
    
 % compute mean for posterior of g(x)
     e       = 1e-6;
     N       = size(Kn, 1);
    if(rank( Kn ) == N)
        invK_f  = Kn \ g_es;
    else
        while(rank(Kn) ~= N)
            e   = e * 10;
            Kn  = Kn + e * eye(N);
            fprintf('singular matrix and e=%.3f',e)
        end
        invK_f  = Kn \ g_es;

    end
    
    mg     = K_testob * invK_f;
%     mg     = posterior_mean (x_es, x_es, g_es, theta);
%     mg     = posteriorMean2(x, x_es, g_es, KernelMatrixInv, l, sigma);
    
 % approximated covariance matrix of y(x) using linearisation
    Kny    = mg * mg' .* Kn;
    
 % compute mean for posterior mean of y(x)
     e     = 1e-6;
     N     = size(Kny, 1);
    if(rank(Kny) == N)
        invK_f  = Kny \ y_es;
    else
        while(rank(Kny) ~= N)
            e   = e * 10;
            Kny = Kny + e * eye(N);
            fprintf('singular matrix and e=%.3f',e)
        end
        invK_f  = Kny \ y_es;
    end
    
 % compute log likelihood 
    T1     = - (y_es' * invK_f) ./ 2.0;
    T2     = - log(det(Kny)) ./ 2.0;
    T3     = - d * log(2 .* pi) ./ 2.0;
    loglikelihood = (T1 + T2 + T3);
    
    end
    
