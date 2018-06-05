function H = computeEntropyGMM1Dquad(means, covs, weights, N)

% Function : Compute entropy of a 1D Gaussian mixture by numerical integration
% ----------------------------------------
% means - array of means of Gaussians in the mixture
% covs - array of covariance matrices of Gaussians in the mixture
% weights - vector of weights of Gaussians in the mixture
% N - (optional) number of st.dev. to include in the integration region (large values can make integration unstable)

% set parameters
if ~exist('N', 'var')
    N = 3;      % number of st.dev. to include in the integration region
end

% set integration boundaries
maxstd = sqrt(covs);
minX   = min(means - N * maxstd, [], 1);
maxX   = max(means + N * maxstd, [], 1);

% compute entropy
% tic
pdfGMM = @(x)GMMpdf(x, means, covs, weights);
% figure 
% plot(x, pdfGMM(x),'b','DisplayName', 'TrueGMM' );
% hold on
integration = @(y) - pdfGMM(y).* log(pdfGMM(y));
H = integral(integration,minX,maxX,'AbsTol',1e-4);
% toc
% entropyFunc2 = @(x)entropyFunc(x, means, maxstd, weights);
% H            = -quad(entropyFunc2, minX, maxX, 1e-4);

end
