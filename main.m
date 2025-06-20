
clear all; clc;
dbstop if error

% Databases to load
db = 'dummy'; % databases are stored on an external hard drive
dbs_to_load = {db};

% DB specific code
switch 1
    case strcmp(db, 'dummy')
        epsilon = 1e-4;  % numerical stability
        clr = [0 0 0];
        mrkr = 'o';
    case strcmp(db, 'Some other db')
end

% Initialize data point storage
x1 = []; x2 = []; x3 = []; x4 = []; S = [];
for db_name = dbs_to_load
    db = db_name{:};
    [a1, a2, a3, a4, A] = getData(db);
    x1 = [x1 a1]; x2 = [x2 a2]; x3 = [x3 a3]; x4 = [x4 a4]; S = [S A];
end

S = S(:).^2;

%% Leave-one-out cross-validation to check accuracy of Kriging durrogate model
LOOCV(x1, x2, x3, x4, S, epsilon, clr, mrkr)

%% Derive Hessian using central finite differences using various step sizes
Hessian_stability_plot = true;
Hessian_components(x1, x2, x3, x4, S, epsilon, Hessian_stability_plot, clr);

%% Derive Hessian using central finite differences using chosen step size
Hessian_stability_plot = false;
H = Hessian_components(x1, x2, x3, x4, S, epsilon, Hessian_stability_plot);

%% Plot required info
% correlation matrix
D = sqrt(diag(H));          % Vector of "standard deviations"
R = H ./ (D * D');          % Element-wise division

% Make figure
figure;
features = {'$E$', '$\sigma_0$', '$\sigma_s$', '$\epsilon_0$'}; % the parameters I'm working with
imagesc(R);
colormap('hsv'); clim([-1 1])
axis equal tight;
xticks(1:length(features));
yticks(1:length(features));
xticklabels(features);
yticklabels(features);
set(gca, 'TickLabelInterpreter', 'latex');
ax = gca;
ax.FontSize = 25;

% Add numeric values
for i = 1:size(H,1)
    for j = 1:size(H,2)
        val = R(i,j);
        text(j, i, sprintf('%.2f', val), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle', ...
             'Color', 'w', ...
             'FontSize', 20, ...
             'FontName', 'Times new roman');
    end
end
