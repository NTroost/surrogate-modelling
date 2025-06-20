function Taylor2expansion(x1, x2, x3, x4, S, H, clr, mrkr)
disp('Running Taylor 2 expansion')

%--- Remove out-of-bounds simulations
inbounds = (x1 >= 0.9 & x1 <= 1.1) & ...
	   (x2 >= 0.9 & x2 <= 1.1) & ...
	   (x3 >= 0.9 & x3 <= 1.1) & ...
	   (x4 >= 0.9 & x4 <= 1.1);

% x1 = x1(inbounds);
% x2 = x2(inbounds);
% x3 = x3(inbounds);
% x4 = x4(inbounds);
mS = S(inbounds);

% Normalization factor
globalmaxS = max(mS(:));

%--- Data preparation
dataPoints = [x1(:)-1, x2(:)-1, x3(:)-1, x4(:)-1];
x0 = [0; 0; 0; 0];

%--- Taylor expansion (no linear terms)
gradS = [0; 0; 0; 0];

T2 = 0.5 * sum((dataPoints * H) .* dataPoints, 2); % Vectorized
errors = T2 - S;

%--- 2D error plots for parameter pairs
paramNames = {'x1', 'x2', 'x3', 'x4'};
plotNum = 1;

figure(104);
for i = 1:3
	for j = i+1:4
        validRows = (dataPoints(:, i) ~= x0(i)) & (dataPoints(:, j) ~= x0(j));
        x = dataPoints(validRows, i);
        y = dataPoints(validRows, j);
        c = errors(validRows) ./ globalmaxS * 100;

        semilogx(sqrt(x.^2 + y.^2) * 100, c, mrkr, 'color', clr);
        setFig; grid on;
        yl = get(gca, 'ylim'); yl = max(abs(yl)); ylim([-yl yl]);
        
        plotNum = plotNum + 1;
    end
end
xlabel(['Distance from origin (%)']);
ylabel('Relative Error T2 - S (%)');

%--- Plot smooth Taylor expansion + simulation points
n = 1001; plot_range = 0.1;
ax = -plot_range + (2*plot_range) * rand(1, n);
ay = -plot_range + (2*plot_range) * rand(1, n);
zs = zeros(size(ax));

newDataPoints = [ [ax(:); ax(:); ax(:); zs(:); zs(:); zs(:)], ...
				  [ay(:); zs(:); zs(:); ax(:); ax(:); zs(:)], ...
				  [zs(:); ay(:); zs(:); ay(:); zs(:); ax(:)], ...
				  [zs(:); zs(:); ay(:); zs(:); ay(:); ay(:)] ];

T22 = 0.5 * sum((newDataPoints * H) .* newDataPoints, 2);

figure;
plotNum = 1;
for i = 1:3
	for j = i+1:4
		validRows = (newDataPoints(:, i) ~= x0(i)) & (newDataPoints(:, j) ~= x0(j));
		validRows1 = (dataPoints(:, i) ~= x0(i)) & (dataPoints(:, j) ~= x0(j));

		x = newDataPoints(validRows, i);
		y = newDataPoints(validRows, j);
		c = T22(validRows);

		x1 = dataPoints(validRows1, i);
		y1 = dataPoints(validRows1, j);
		c1 = S(validRows1);

		subplot(2, 3, plotNum);
		vectorSurf(x, y, c); hold on
		plot3(x1, y1, c1, 'or');
		setFig; grid on;
		xlabel(paramNames{i});
		ylabel(paramNames{j});
		title(['Taylor Expansion: ', paramNames{i}, ' vs ', paramNames{j}]);

		plotNum = plotNum + 1;
	end
end

%--- Error along minimum degenerate directions
figure(106); setFig;
plotNum = 1;
for idx = 1:3
    for jdx = idx+1:4
        validRows = (dataPoints(:, idx) ~= x0(idx)) & (dataPoints(:, jdx) ~= x0(jdx));
        x = dataPoints(validRows, idx);
        y = dataPoints(validRows, jdx);
        err = errors(validRows) ./ globalmaxS * 100;

        F = scatteredInterpolant(x, y, err, 'natural', 'none');

        Hii = H(idx, idx);
        Hjj = H(jdx, jdx);

        n_samples = 200;
        pm = sign(H(idx,jdx));
        x_line = linspace(-0.1, 0.1, n_samples);
        y_line = -pm*(sqrt(Hii)/sqrt(Hjj)) * x_line;
        plot_line = sign(x_line) .* sqrt(x_line.^2 + y_line.^2);
        err_line = F(x_line, y_line);

        plot(plot_line * 100, err_line, 'Color', clr, ...
            'DisplayName', [paramNames{idx} ' vs ' paramNames{jdx}], 'LineWidth', 1.5);
        % figure(2); subplot(1, 3, plotNum);
        % plot(x_line, y_line, 'k')

        plotNum = plotNum + 1;
    end
end
xlabel('Distance along degenerate direction (%)');
ylabel('Error (Taylor2 - actual)');
legend('Location', 'best');
title('Error along minimum-degenerate directions');
setFig;
end
