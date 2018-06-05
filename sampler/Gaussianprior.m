function gp_prior=Gaussianprior(lntheta,mean, variance)
% Function: compute Gaussian prior
    gp_prior=mvnpdf(lntheta, mean, variance);
end