function [out1,out2,out3,out4,objective_function_values] = getData(db)
% Sample data points (X, y, z, q) and corresponding objective function values S

switch 1
    case strcmp(db, 'dummy')
        % I always have a dummy data base for testing anyway
        n = 500; % amount of discrete points
        x1 = normrnd(0,.05,[1 n]); % Random variables
        x2 = normrnd(0,.05,[1 n]);
        x3 = normrnd(0,.05,[1 n]);
        x4 = normrnd(0,0.05,[1 n]);

        Hd = [10 50 100 500]; % Hessian values of objective function
        Sf = @(s1, s2, s3, s4) ... % dummy objective function
            0.5*Hd(1)*s1.^2 + 0.5*Hd(2)*s2.^2 + 0.5*Hd(3)*s3.^2 + 0.5*Hd(4)*s4.^2 + ...
            sqrt(Hd(1)*Hd(2))*s1.*s2 + ... % create degeneracy (for testing)
            sqrt(Hd(1)*Hd(3))*s1.*s3 + ...
            sqrt(Hd(1)*Hd(4))*s1.*s4 + ...
            sqrt(Hd(2)*Hd(3))*s2.*s3 + ...
            sqrt(Hd(2)*Hd(4))*s2.*s4 + ...
            sqrt(Hd(3)*Hd(4))*s3.*s4;
        objective_function_values = Sf(x1, x2, x3, x4);

        out1 = x1+1; out2 = x2+1; out3 = x3+1; out4 = x4+1; % variables are normalized and centered around 1

    otherwise
        % Here is where you would load the actual databases from
end