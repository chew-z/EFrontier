function [sharpe, sigma, mu] = minvar(Mu, Cov)
% minvar returns minimum variance portfolio using pinv(Covariance matrix)
% this is analytical solution as good as covariance and mean estimator
%	
	numalphas = length(Mu);
	
	iS = pinv(Cov); % inverse covariance matrix with pinv
    % set weigths  ( inv(Cov) * 1 / 1'*inv(Cov)*1
    weights = (iS * ones(numalphas,1)) / (ones(1,numalphas) * iS * ones(numalphas,1));

    sigma = sqrt(weights' *Cov* weights);
    mu = Mu * weights;
    sharpe = mu / sigma;
end