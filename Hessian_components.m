function H = Hessian_components(x1, x2, x3, x4, S, epsilon, plt, clr)
disp('Running Hessian components')

% Combine x1, x2, x3, and x4 into a single matrix for training data
dataPoints = [x1(:), x2(:), x3(:), x4(:)];

% Define a function for the covariance model and initial hyperparameters
% covFunc = @(d, theta) exp(-theta(1) * d.^2); % Exponential kernel
covFunc = @(d, theta) exp(-d.^2 / (2 * theta(1)^2)); % Gaussian kernel
theta0 = .01;

% Define the log-likelihood function for optimization
logLikelihood = @(theta) -logLikelihoodFunction(dataPoints, S, covFunc, theta, epsilon);

% Optimize hyperparameters
options = optimset('MaxIter', 1000, 'TolX', 1e-12);
thetaHat = fminunc(logLikelihood, theta0, options);

% Define the minimization function for kriging prediction
minfun = @(s) krigingPredict([s(1), s(2), s(3), s(4)], dataPoints, S, covFunc, thetaHat, epsilon);

% Find the solution using fmincon
% Initial guess, bounds, and objective function
optionsfmincon = optimoptions(@fmincon, ...
    'OptimalityTolerance', 1e-8, 'ConstraintTolerance', 1e-8, ...
    'MaxFunctionEvaluations', 10000, ...
    'MaxIterations', 10000);
lb = [1 1 1 1]; % I know the solution is 1 ...
ub = [1 1 1 1]; % But in actual optimisation this would be different

% Call fmincon
x0 = [1 1 1 1]; % Same thing
[sol, fval] = fmincon(minfun, x0, [], [], [], [], lb, ub, [], optionsfmincon);

% Update dataPoints to center around solution
x1 = x1 - sol(1); x2 = x2 - sol(2); x3 = x3 - sol(3); x4 = x4 - sol(4);

% Redefine dataPoints and log-likelihood after finding solution
% and optimize hyperparameters again with updated data
dataPoints = [x1(:), x2(:), x3(:), x4(:)];
logLikelihood = @(theta) -logLikelihoodFunction(dataPoints, S, covFunc, theta, epsilon);
thetaHat = fminunc(logLikelihood, theta0, options);

% Base prediction at (0,0,0,0)
[S_pred, ~] = krigingPredict([0, 0, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);

% Initialize plot
if plt
    % Plot for various step sizes
    dd = logspace(log10(10^-8), log10(.1), 101);
else
    % I know this is not a great solution
    dd = 2e-4;
end

% Initialize storage for Hessian components
dslopex1 = zeros(1, length(dd));
dslopex2 = zeros(1, length(dd));
dslopex3 = zeros(1, length(dd));
dslopex4 = zeros(1, length(dd));
dslopex12 = zeros(1, length(dd));
dslopex13 = zeros(1, length(dd));
dslopex14 = zeros(1, length(dd));
dslopex23 = zeros(1, length(dd));
dslopex24 = zeros(1, length(dd));
dslopex34 = zeros(1, length(dd));

% Hessian
H = NaN;

for i = 1:length(dd)
    dx = dd(i);

    % Compute constants
    dx2 = dx^2;
    factor12dx2 = 12 * dx2;
    factor4dx2 = 4 * dx2;

    % Second partial derivative w.r.t. x1 
    [S_pred_m2dx, ~] = krigingPredict([-2*dx, 0, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx, ~] = krigingPredict([-dx, 0, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_pdx, ~] = krigingPredict([dx, 0, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_p2dx, ~] = krigingPredict([2*dx, 0, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);

    d2S_dx1 = (-S_pred_m2dx + 16 * S_pred_mdx - 30 * S_pred + 16 * S_pred_pdx - S_pred_p2dx) / factor12dx2;
    dslopex1(i) = d2S_dx1;

    %  Second partial derivative w.r.t. x2 
    [S_pred_m2dx, ~] = krigingPredict([0, -2*dx, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx, ~] = krigingPredict([0, -dx, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_pdx, ~] = krigingPredict([0, dx, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_p2dx, ~] = krigingPredict([0, 2*dx, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);

    d2S_dx2 = (-S_pred_m2dx + 16 * S_pred_mdx - 30 * S_pred + 16 * S_pred_pdx - S_pred_p2dx) / factor12dx2;
    dslopex2(i) = d2S_dx2;

    %  Second partial derivative w.r.t. x3 
    [S_pred_m2dx, ~] = krigingPredict([0, 0, -2*dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx, ~] = krigingPredict([0, 0, -dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_pdx, ~] = krigingPredict([0, 0, dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_p2dx, ~] = krigingPredict([0, 0, 2*dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);

    d2S_dx3 = (-S_pred_m2dx + 16 * S_pred_mdx - 30 * S_pred + 16 * S_pred_pdx - S_pred_p2dx) / factor12dx2;
    dslopex3(i) = d2S_dx3;

    %  Second partial derivative w.r.t. x4 
    [S_pred_m2dx, ~] = krigingPredict([0, 0, 0, -2*dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx, ~] = krigingPredict([0, 0, 0, -dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_pdx, ~] = krigingPredict([0, 0, 0, dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_p2dx, ~] = krigingPredict([0, 0, 0, 2*dx], dataPoints, S, covFunc, thetaHat, epsilon);

    d2S_dx4 = (-S_pred_m2dx + 16 * S_pred_mdx - 30 * S_pred + 16 * S_pred_pdx - S_pred_p2dx) / factor12dx2;
    dslopex4(i) = d2S_dx4;

    %  Mixed partial derivatives 
    % d2S_dx1dx2
    Hcrit = sqrt(d2S_dx1*d2S_dx2);
    [S_pred_pdx1_pdx2, ~] = krigingPredict([dx, dx, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_pdx1_mdx2, ~] = krigingPredict([dx, -dx, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx1_pdx2, ~] = krigingPredict([-dx, dx, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx1_mdx2, ~] = krigingPredict([-dx, -dx, 0, 0], dataPoints, S, covFunc, thetaHat, epsilon);

    d2S_dx1dx2 = (S_pred_pdx1_pdx2 - S_pred_pdx1_mdx2 - S_pred_mdx1_pdx2 + S_pred_mdx1_mdx2) / factor4dx2;
    dslopex12(i) = d2S_dx1dx2 / Hcrit;

    % d2S_dx1dx3
    Hcrit = sqrt(d2S_dx1*d2S_dx3);
    [S_pred_pdx1_pdx3, ~] = krigingPredict([dx, 0, dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_pdx1_mdx3, ~] = krigingPredict([dx, 0, -dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx1_pdx3, ~] = krigingPredict([-dx, 0, dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx1_mdx3, ~] = krigingPredict([-dx, 0, -dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);

    d2S_dx1dx3 = (S_pred_pdx1_pdx3 - S_pred_pdx1_mdx3 - S_pred_mdx1_pdx3 + S_pred_mdx1_mdx3) / factor4dx2;
    dslopex13(i) = d2S_dx1dx3 / Hcrit;

    % d2S_dx1dx4
    Hcrit = sqrt(d2S_dx1*d2S_dx4);
    [S_pred_pdx1_pdx4, ~] = krigingPredict([dx, 0, 0, dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_pdx1_mdx4, ~] = krigingPredict([dx, 0, 0, -dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx1_pdx4, ~] = krigingPredict([-dx, 0, 0, dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx1_mdx4, ~] = krigingPredict([-dx, 0, 0, -dx], dataPoints, S, covFunc, thetaHat, epsilon);

    d2S_dx1dx4 = (S_pred_pdx1_pdx4 - S_pred_pdx1_mdx4 - S_pred_mdx1_pdx4 + S_pred_mdx1_mdx4) / factor4dx2;
    dslopex14(i) = d2S_dx1dx4 / Hcrit;

    % d2S_dx2dx3
    Hcrit = sqrt(d2S_dx2*d2S_dx3);
    [S_pred_pdx2_pdx3, ~] = krigingPredict([0, dx, dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_pdx2_mdx3, ~] = krigingPredict([0, dx, -dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx2_pdx3, ~] = krigingPredict([0, -dx, dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx2_mdx3, ~] = krigingPredict([0, -dx, -dx, 0], dataPoints, S, covFunc, thetaHat, epsilon);

    d2S_dx2dx3 = (S_pred_pdx2_pdx3 - S_pred_pdx2_mdx3 - S_pred_mdx2_pdx3 + S_pred_mdx2_mdx3) / factor4dx2;
    dslopex23(i) = d2S_dx2dx3 / Hcrit;

    % d2S_dx2dx4
    Hcrit = sqrt(d2S_dx2*d2S_dx4);
    [S_pred_pdx2_pdx4, ~] = krigingPredict([0, dx, 0, dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_pdx2_mdx4, ~] = krigingPredict([0, dx, 0, -dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx2_pdx4, ~] = krigingPredict([0, -dx, 0, dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx2_mdx4, ~] = krigingPredict([0, -dx, 0, -dx], dataPoints, S, covFunc, thetaHat, epsilon);

    d2S_dx2dx4 = (S_pred_pdx2_pdx4 - S_pred_pdx2_mdx4 - S_pred_mdx2_pdx4 + S_pred_mdx2_mdx4) / factor4dx2;
    dslopex24(i) = d2S_dx2dx4 / Hcrit;

    % d2S_dx3dx4
    Hcrit = sqrt(d2S_dx3*d2S_dx4);
    [S_pred_pdx3_pdx4, ~] = krigingPredict([0, 0, dx, dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_pdx3_mdx4, ~] = krigingPredict([0, 0, dx, -dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx3_pdx4, ~] = krigingPredict([0, 0, -dx, dx], dataPoints, S, covFunc, thetaHat, epsilon);
    [S_pred_mdx3_mdx4, ~] = krigingPredict([0, 0, -dx, -dx], dataPoints, S, covFunc, thetaHat, epsilon);

    d2S_dx3dx4 = (S_pred_pdx3_pdx4 - S_pred_pdx3_mdx4 - S_pred_mdx3_pdx4 + S_pred_mdx3_mdx4) / factor4dx2;
    dslopex34(i) = d2S_dx3dx4 / Hcrit;

    % Print in case only one dd, show Hessian
    if isscalar(dd)
        % Again not pretty but it works
        H = [d2S_dx1 d2S_dx1dx2 d2S_dx1dx3 d2S_dx1dx4;
            d2S_dx1dx2 d2S_dx2 d2S_dx2dx3 d2S_dx2dx4;
            d2S_dx1dx3 d2S_dx2dx3 d2S_dx3 d2S_dx3dx4
            d2S_dx1dx4 d2S_dx2dx4 d2S_dx3dx4 d2S_dx4];

        disp(H);
    end
end

% Plotting
if plt
    figure;
    subplot(4, 4, 1); loglog(dd, dslopex1, '.b'); title('Second Derivative w.r.t. x1'); ylim([.01 100000]); grid on;
    subplot(4, 4, 6); loglog(dd, dslopex2, '.b'); title('Second Derivative w.r.t. x2'); ylim([.01 100000]); grid on;
    subplot(4, 4, 11); loglog(dd, dslopex3, '.b'); title('Second Derivative w.r.t. x3'); ylim([.01 100000]); grid on;
    subplot(4, 4, 16); loglog(dd, dslopex4, '.b'); title('Second Derivative w.r.t. x4'); ylim([.01 100000]); grid on;

    subplot(4, 4, 2); semilogx(dd, abs(dslopex12), '.k'); title('Mixed (critical) Derivative w.r.t. x1 and x2'); grid on;  ylim([-0.5 1.5]);
    subplot(4, 4, 3); semilogx(dd, abs(dslopex13), '.k'); title('Mixed (critical) Derivative w.r.t. x1 and x3'); grid on;  ylim([-0.5 1.5]);
    subplot(4, 4, 4); semilogx(dd, abs(dslopex14), '.k'); title('Mixed (critical) Derivative w.r.t. x1 and x4'); grid on;  ylim([-0.5 1.5]);
    subplot(4, 4, 7); semilogx(dd, abs(dslopex23), '.k'); title('Mixed (critical) Derivative w.r.t. x2 and x3'); grid on;  ylim([-0.5 1.5]);
    subplot(4, 4, 8); semilogx(dd, abs(dslopex24), '.k'); title('Mixed (critical) Derivative w.r.t. x2 and x4'); grid on;  ylim([-0.5 1.5]);
    subplot(4, 4, 12); semilogx(dd, abs(dslopex34), '.k'); title('Mixed (critical) Derivative w.r.t. x3 and x4'); grid on;  ylim([-0.5 1.5]);

    subplot(4, 4, 5); semilogx(dd, abs(dslopex12), '.k'); title('Mixed (critical) Derivative w.r.t. x1 and x2'); grid on;  ylim([-0.5 1.5]);
    subplot(4, 4, 9); semilogx(dd, abs(dslopex13), '.k'); title('Mixed (critical) Derivative w.r.t. x1 and x3'); grid on;  ylim([-0.5 1.5]);
    subplot(4, 4, 13); semilogx(dd, abs(dslopex14), '.k'); title('Mixed (critical) Derivative w.r.t. x1 and x4'); grid on;  ylim([-0.5 1.5]);
    subplot(4, 4, 10); semilogx(dd, abs(dslopex23), '.k'); title('Mixed (critical) Derivative w.r.t. x2 and x3'); grid on;  ylim([-0.5 1.5]);
    subplot(4, 4, 14); semilogx(dd, abs(dslopex24), '.k'); title('Mixed (critical) Derivative w.r.t. x2 and x4'); grid on;  ylim([-0.5 1.5]);
    subplot(4, 4, 15); semilogx(dd, abs(dslopex34), '.k'); title('Mixed (critical) Derivative w.r.t. x3 and x4'); grid on;  ylim([-0.5 1.5]);

    figure;
    ddq = logspace(log10(min(dd)), log10(max(dd)), 1001);
    semilogx(ddq*100, interp1(dd, (gradient(dslopex1)./gradient(log10(dd))), ddq, 'pchip'), 'k', 'LineWidth', 2, 'color', clr); hold on; grid on;
    semilogx(ddq*100, interp1(dd, (gradient(dslopex2)./gradient(log10(dd))), ddq, 'pchip'), 'k', 'LineWidth', 2, 'color', clr);
    semilogx(ddq*100, interp1(dd, (gradient(dslopex3)./gradient(log10(dd))), ddq, 'pchip'), 'k', 'LineWidth', 2, 'color', clr);
    semilogx(ddq*100, interp1(dd, (gradient(dslopex4)./gradient(log10(dd))), ddq, 'pchip'), 'k', 'LineWidth', 2, 'color', clr);

    semilogx(ddq*100, interp1(dd, (gradient(dslopex12)./gradient(log10(dd))), ddq, 'pchip'), 'k', 'LineWidth', 2, 'color', clr);
    semilogx(ddq*100, interp1(dd, (gradient(dslopex13)./gradient(log10(dd))), ddq, 'pchip'), 'k', 'LineWidth', 2, 'color', clr);
    semilogx(ddq*100, interp1(dd, (gradient(dslopex14)./gradient(log10(dd))), ddq, 'pchip'), 'k', 'LineWidth', 2, 'color', clr);
    semilogx(ddq*100, interp1(dd, (gradient(dslopex23)./gradient(log10(dd))), ddq, 'pchip'), 'k', 'LineWidth', 2, 'color', clr);
    semilogx(ddq*100, interp1(dd, (gradient(dslopex24)./gradient(log10(dd))), ddq, 'pchip'), 'k', 'LineWidth', 2, 'color', clr);
    semilogx(ddq*100, interp1(dd, (gradient(dslopex34)./gradient(log10(dd))), ddq, 'pchip'), 'k', 'LineWidth', 2, 'color', clr);
end

    function q = vectorKrigingPredict(x1, x2, x3, x4, dataPoints, S, covFunc, thetaHat, epsilon)

        for tmp = 1:length(x1)
            q(tmp) = krigingPredict([x1(tmp), x2(tmp), x3(tmp), x4(tmp)], dataPoints, S, covFunc, thetaHat, epsilon);
        end
    end
end

