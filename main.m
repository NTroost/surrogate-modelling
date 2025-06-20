
clear all; clc;
dbstop if error

% Databases to load
db = 'TiVO'; % example, could also be 'AlVO', 'BrVO', 'TiVO'
D = 2;

switch db
    case 'StVO'
        % Code to load StRO
        epsilon = 1e-4;
        clr = [0 0 0];
        mrkr = 'o';
        F = 2200; fac = 1;
    case 'AlVO'
        % Code to load AlRO
        epsilon = 1e-4;
        clr = [0 0.6 0];
        mrkr = 'x';
        F = 1390; fac = 1;
    case 'BrVO'
        % Code to load BrRO
        epsilon = 1e-6;
        clr = [0 0 0.8];
        mrkr = '^';
        F = 1517; fac = 1;
    case 'TiVO'
        % Code to load TiRO
        epsilon = 1e-5;
        clr = [0.8 0 0];
        mrkr = 'v';
        F = 4807; fac = 14 / 12;
    otherwise
        error('Unknown database selected.');
end
dbs_to_load = {db};

% Initialize data point storage
x1 = []; x2 = []; x3 = []; x4 = []; S = [];
for db_name = dbs_to_load
    % db = ['31_databases\' db_name{:}];
    db = db_name{:};
    [a1, a2, a3, a4, a, d, PUhn, UTS, sy] = getData(db, 4);
    x1 = [x1 a1]; x2 = [x2 a2]; x3 = [x3 a3]; x4 = [x4 a4]; S = [S a];
end

S = S(:).^2;

% %%
% Hessian_stability_plot = true;
% Hessian_componentsVoce(x1, x2, x3, x4, S, epsilon, Hessian_stability_plot, clr);
% 
% %%
% LOOCVVoce(x1, x2, x3, x4, S, epsilon, clr, mrkr)

%%
disp(median(UTS)/fac)
kgf = F / 9.81; d = d/1000;
HB = (2 * kgf) ./ (pi * D * (D - sqrt(D^2 - d.^2)));
figure(50); plot3(PUhn, UTS, HB, mrkr, 'color', clr); setFig

%%
Hessian_stability_plot = false;
H = Hessian_componentsVoce(x1, x2, x3, x4, S, epsilon, Hessian_stability_plot);

% % H is your Hessian matrix (n x n)
% D = sqrt(diag(H));          % Vector of "standard deviations"
% R = H ./ (D * D');          % Element-wise division
% 
% R(R>1.001) = 0.99;
% R(R<-1.001) = -.99;
% 
% figure;
% features = {'$E$', '$\sigma_0$', '$\sigma_s$', '$\epsilon_0$'};
% imagesc(R);
% colormap('hsv'); clim([-1 1])
% axis equal tight;
% xticks(1:length(features));
% yticks(1:length(features));
% xticklabels(features);
% yticklabels(features);
% set(gca, 'TickLabelInterpreter', 'latex');
% ax = gca;
% ax.FontSize = 25;
% 
% % Add numeric values
% for i = 1:size(H,1)
%     for j = 1:size(H,2)
%         val = R(i,j);
%         text(j, i, sprintf('%.2f', val), ...
%              'HorizontalAlignment', 'center', ...
%              'VerticalAlignment', 'middle', ...
%              'Color', 'w', ...
%              'FontSize', 20, ...
%              'FontName', 'Times new roman');
%     end
% end
% 
% figure;
% colormap('hsv'); clim([-1 1])
% colorbar;
% ax = gca;
% ax.FontSize = 25;
% 
% clc
% for i = 1:size(H,1)
%     disp(H(i,i));
% end
% 
% keyboard

%%
Taylor2Voce(x1, x2, x3, x4, S, H, clr, mrkr)
keyboard

%%
for fign = [3 4]
    figure(fign)
    xlabel('');
    xlabel('');
    xlabel('');

    ax = gca;
    ax.FontSize = 20;

    if fign == 3
        ylim([-1e4 1e4])
        ax.YTick = -1e4:0.25e4:1e4;
        xlim([1e-6 10])
        ax.XTick = logspace(log10(1e-6), log10(10), 8);
    elseif fign == 4
        ylim([-25 25])
        ax.YTick = -25:5:25;
        xlim([1e-2 10])
    end

    % Get the current axes handle (or specify the axes if known)
    ax = gca;

    % Get all line objects (or plot children) in the axes
    lines = findall(ax, 'Type', 'Line');

    % Shuffle the order
    shuffled_indices = randperm(length(lines));
    shuffled_lines = lines(shuffled_indices);

    % Reorder the children of the axes
    ax.Children = shuffled_lines;
end

%%
for fign = [104 106]
    figure(fign)
    legend('off')
    xlabel('');
    ylabel('');
    title('');

    ax = gca;
    ax.FontSize = 20;

    if fign == 104
        ylim([-50 50])
        ax.YTick = -50:25:50;
        xlim([1e-2 10])

    elseif fign == 106
        xlim([-10 10])
    end

    % Get the current axes handle (or specify the axes if known)
    ax = gca;

    % Get all line objects (or plot children) in the axes
    lines = findall(ax, 'Type', 'Line');

    % Shuffle the order
    shuffled_indices = randperm(length(lines));
    shuffled_lines = lines(shuffled_indices);

    % Reorder the children of the axes
    ax.Children = shuffled_lines;
end
