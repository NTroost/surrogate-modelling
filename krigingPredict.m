function [S_pred, sigma_pred] = krigingPredict(newPoint, dataPoints, S, covFunc, theta, varargin)
% Function to predict at a new point with uncertainty
% Optionally, a numerical stability term (default: 1e-6) can be provided.

% Set default numerical stability value
epsilon = 1e-2;

% If an additional argument is provided, use it as epsilon
if ~isempty(varargin)
    epsilon = varargin{1};
end

% Calculate covariance vectors
d_new = pdist2(dataPoints, newPoint); % Distance from training points to the new point
K_new = covFunc(d_new, theta); % Covariance between new point and training points
K = covFunc(pdist2(dataPoints, dataPoints), theta); % Covariance matrix of training points
K = K + eye(size(K)) * epsilon; % Numerical stability

% Perform the Kriging prediction
L = chol(K, 'lower'); % Cholesky decomposition
alpha = L'\(L\S); % Solve L * alpha = S
S_pred = K_new' * alpha; % Predictive mean

% Calculate the variance (uncertainty) at the new point
v = L \ K_new; % Solve L * v = K_new
sigma_pred = covFunc(0, theta) - v' * v; % Predictive variance
end