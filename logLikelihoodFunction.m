function ll = logLikelihoodFunction(dataPoints, S, covFunc, theta, epsilon)
% Function to compute the log-likelihood

n = size(dataPoints, 1);
K = covFunc(pdist2(dataPoints, dataPoints), theta); % Covariance matrix
K = K + eye(n) * epsilon; % Add a small value for numerical stability
L = chol(K, 'lower');
alpha = L'\(L\S); % Solve L * alpha = S
ll = -0.5 * S' * alpha - sum(log(diag(L))) - n/2 * log(2*pi);
end