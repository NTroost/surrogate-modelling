function LOOCV(x1, x2, x3, x4, S, epsilon, clr, mrkr)
disp('Running LOOCV')

% Filter to sim bounds
inds = x1 < 0.90 | x1 > 1.1 | ...
    x2 < 0.90 | x2 > 1.1 | ...
    x3 < 0.90 | x3 > 1.1 | ...
    x4 < 0.90 | x4 > 1.1;

disp(['Deleted sims: ' num2str(sum(inds))]);
x1(inds) = []; x2(inds) = []; x3(inds) = []; x4(inds) = []; S(inds) = [];
nS = S; normaliser = max(nS);

% Combine x1, x2, x3, and x4 into a single matrix for training data
dataPoints = [x1(:), x2(:), x3(:), x4(:)];

% Define a function for the covariance model and initial hyperparameters
% covFunc = @(d, theta) exp(-theta(1) * d.^2); % Exponential kernel
covFunc = @(d, theta) exp(-d.^2 / (2 * theta(1)^2)); % Gaussian kernel
theta0 = [.01];

% Optimize hyperparameters
options = optimset('MaxIter', 1000, 'TolX', 1e-12);

%% Leave-One-Out Cross-Validation (LOOCV) for n Random Points
% Occasionally I dont want to do this for all datapoints, so I choose n
% random ones
n = length(S); % Number of random points for cross-validation
numData = length(S);
randIndices = randperm(numData, n); % Select n random indices

% Init results
distance = zeros(n, 1);
errors = zeros(n, 1);

for i = 1:n
    idx = randIndices(i); % Pick a random index
    disp(['Sim: ' num2str(i) ' / ' num2str(n)]);

    % Compute distance
    distance(i) = norm(dataPoints(idx,:)-1);

    % Remove the selected point
    X_train = dataPoints;
    Y_train = S;
    X_train(idx, :) = []; % Remove i-th row
    Y_train(idx) = [];

    % Define the log-likelihood function for optimization
    logLikelihood = @(theta) -logLikelihoodFunction(X_train, Y_train, covFunc, theta, epsilon);
    thetaHat = fminunc(logLikelihood, theta0, options);

    % Predict the left-out point
    Y_pred = krigingPredict(dataPoints(idx,:), X_train, Y_train, covFunc, thetaHat, epsilon);

    % Compute prediction error
    errors(i) = (S(idx) - Y_pred)/normaliser*100;
end

%%
% Compute Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
MAE = mean(errors);
RMSE = sqrt(mean(errors.^2));

% Display results
clc
fprintf('LOO-CV Results for %d Random Points:\n', n);
fprintf('Mean Absolute Error (MAE): %.4f\n', MAE);
fprintf('Root Mean Squared Error (RMSE): %.4f\n', RMSE);

% Plot in steps so I can reorder later when multiple DBs are involved
% (makes sense for paper)
figure;
step = floor(0.05 * length(errors));
for i = 1:step:length(errors)
    idx_end = min(i + step - 1, length(errors));
    semilogx(distance(i:idx_end)*100, errors(i:idx_end), mrkr, 'color', clr); hold on;
end
end
