function FITBOacq(objective, lb, hb, option)

var_noise       = option.var_noise;     % variance of true output noise level 
nInitialSamples = option.nInitialSamples;    % number of initial observation data points
ess             = option.ess;     % use elliptical slice sampler (1) or slice sampler (0) 
N_hypsample     = option.N_hypsample; % number of hyperparameter samples
N_evaluation    = option.N_evaluation;% number of evaluations
N_seed          = option.N_seed;      % number of random initialisations
useMM           = option.useMM;       % use MM for entropy of GMM
d               = length(lb);         % dimension of data
e               = exp(1);             % exponential term of power 1         

warning('off','all')

%%%%%%%%%% STORE DATA %%%%%%%%%%
% minimiser and minimum of the obj function predicted by FITBO
minimiser_FITBO             = zeros(N_evaluation,d,N_seed);
min_predict_FITBO           = zeros(N_evaluation,N_seed);
% next evaluation suggested by FITBO
next_eval_FITBO             = zeros(N_evaluation,N_seed);
next_eval_location_FITBO    = zeros(N_evaluation,d,N_seed);
eta_values_FITBO            = zeros(N_evaluation,N_seed,N_hypsample);

%% %%%%%%%%%% FITBO ALGORITHM %%%%%%%%%%
for s = 10 : N_seed
    
    % initilise random number generator
    seed        = s   ;
    rng(seed,'twister');

    % randomly generate 3 observation data points
    Xob         = lhsu(lb, hb, nInitialSamples);
    Yob         = zeros(nInitialSamples, 1);
    for i2 = 1 : nInitialSamples
        Yob(i2) = objective(Xob(i2,:)) - randn(1)* sqrt(var_noise);
    end     
    
    x_es        = Xob;
    y_es        = Yob;    

    guesses = x_es;
    guessvals = y_es;
    
    % intial guess for the parameters of the prior distribution of log(yminob - eta) 
    mean_ln_yminob_minus_eta = log(1.0);
    var_ln_yminob_minus_eta  = 0.1;      
    % obtain log(hyperparameters) samples
    ln_hypsamples = samplehyperparameters(x_es,y_es, N_hypsample,mean_ln_yminob_minus_eta,var_ln_yminob_minus_eta,ess);
    % estimate mean and variance of the prior distribution of log(yminob - eta) from samples 
    [mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta] = normfit(ln_hypsamples(:, d + 3));
    % obtain samples of hyperparameters including yminob - eta 
    hypsamples               = exp(ln_hypsamples);    
    l = 1./hypsamples(1:N_hypsample,1:d).^2; 
    sigma = hypsamples(1:N_hypsample, d+1);
    sigma0 = hypsamples(1:N_hypsample,d+2);
    
    % precompute inversion of Gram matrix for all hyperparameters
    KernelMatrixInv = cell(1, N_hypsample);
    for j = 1 : N_hypsample
        KernelMatrix = computeKmm(x_es, l(j,:), sigma(j), sigma0(j));
        KernelMatrixInv{ j } = chol2invchol(KernelMatrix);
    end
    
    for k = 1 : N_evaluation

        rng(seed + 100 * k, 'twister');
       
        % find the minimum value observed
        yminob     = min(y_es);
        eta    = yminob - hypsamples(1:N_hypsample, d + 3);

        % Use moment matching for GMM entropy 
        if useMM == 1
            % optimise FITBOMM acquisition function via global optimiser to obtain the next evaluation point
            AcquisitionMM = @(x) FITBOMM_acqu_function(x, x_es, y_es, KernelMatrixInv, l, sigma, sigma0, eta, N_hypsample);        
            x_next      = globalMaximization(AcquisitionMM, lb', hb', guesses);
        
        % Use quadrature for GMM entropy 
        else
            % optimise FITBO acquisition function via global optimiser to obtain the next evaluation point
            Acquisition = @(x) FITBO_acqu_function(x, x_es, y_es, KernelMatrixInv, l, sigma, sigma0, eta, N_hypsample);        
            x_next      = Optimization(Acquisition, lb, hb, guesses);
        end
        
        y_next      = objective(x_next) + sqrt(var_noise) .* randn(size(x_next, 1), 1);
                      
        % augment the observation data with with the next sample
        x_es        = [x_es; x_next];
        y_es        = [y_es; y_next];
        
        % store the next evalution point
        next_eval_FITBO(k, s)           = y_next;
        next_eval_location_FITBO(k,:,s) = x_next;
        
        % resample hyperparamters using augmented observation data set  
        ln_hypsamples = samplehyperparameters(x_es,y_es, N_hypsample,mean_ln_yminob_minus_eta,var_ln_yminob_minus_eta,ess);
        [mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta] = normfit(ln_hypsamples(:, d+3));
        hypsamples    = exp(ln_hypsamples);   
        l = 1./hypsamples(1:N_hypsample,1:d).^2; 
        sigma = hypsamples(1:N_hypsample, d+1);
        sigma0 = hypsamples(1:N_hypsample,d+2);
        
        % store the eta samples
        eta_values_FITBO(k, s, :) = yminob - hypsamples(:, d + 3);
        
        % Precompute the Gram matrix for all hyperparameter samples    
        KernelMatrixInv = cell(1, N_hypsample);
        for j = 1 : N_hypsample
            KernelMatrix = computeKmm(x_es, l(j,:), sigma(j), sigma0(j));
            KernelMatrixInv{ j } = chol2invchol(KernelMatrix);
        end
        
        % optimise the posterior mean of the updated GP to find the best guess 
        f = @(x) posteriorMean(x, x_es, y_es, KernelMatrixInv, l, sigma);
        gf = @(x) gradientPosteriorMean(x, x_es, y_es, KernelMatrixInv, l, sigma);
        x_minimiser = globalOptimization(f, gf,lb, hb, guesses);
        guesses = [guesses;x_minimiser];
      
        % store the minimiser and minimum prediced 
        minimiser_FITBO(k, :, s) = x_minimiser;
        min_predict_FITBO(k, s)  = objective(x_minimiser);
        
        disp(['Seed=' num2str(s) ': eval= ' num2str(k) ';guess=' num2str(x_minimiser) ';val=' num2str(min_predict_FITBO(k, s)) ])
        if mod(k, 10) == 0
            if useMM == 1
                save('minimiser_FITBOMM', 'minimiser_FITBO')
                save('min_predict_FITBOMM', 'min_predict_FITBO')
                save('eta_values_FITBOMM', 'eta_values_FITBO')
                save('next_eval_FITBOMM', 'next_eval_FITBO')
                save('next_eval_location_FITBOMM', 'next_eval_location_FITBO')
            else
                save('minimiser_FITBO', 'minimiser_FITBO')
                save('min_predict_FITBO', 'min_predict_FITBO')
                save('eta_values_FITBO', 'eta_values_FITBO')
                save('next_eval_FITBO', 'next_eval_FITBO')
                save('next_eval_location_FITBO', 'next_eval_location_FITBO')
            end
        end
        
    end

end
end