function [out1,out2,out3,out4,objective_function_values] = getData(db)
% Sample data points (x, y, z, q) and corresponding objective function values S

switch 1
    case strcmp(db, 'dummy')
        % I always have a dummy data base for testing anyway
        n = 50; % amount of discrete points
        % Make pairs, fine for Hessian
        x1 = [rv(n); rv(n); rv(n); zv(n); zv(n); zv(n)]; % Random variables
        x2 = [rv(n); zv(n); zv(n); rv(n); rv(n); zv(n)]; % Random variables
        x3 = [zv(n); rv(n); zv(n); rv(n); zv(n); rv(n)]; % Random variables
        x4 = [zv(n); zv(n); rv(n); zv(n); rv(n); rv(n)]; % Random variables

        Hd = [10 20 30 40]; % Hessian values of objective function
        Sf = @(s1, s2, s3, s4) ... % dummy objective function
            0.5*Hd(1)*s1.^2 + 0.5*Hd(2)*s2.^2 + 0.5*Hd(3)*s3.^2 + 0.5*Hd(4)*s4.^2 + ...
            0.5*sqrt(Hd(1)*Hd(2))*s1.*s2 + ... % create degeneracy (for testing)
            sqrt(Hd(1)*Hd(3))*s1.*s3 + ...
            -0.5*sqrt(Hd(1)*Hd(4))*s1.*s4 + ...
            -sqrt(Hd(2)*Hd(3))*s2.*s3 + ...
            0.5*sqrt(Hd(2)*Hd(4))*s2.*s4 + ...
            -sqrt(Hd(3)*Hd(4))*s3.*s4;
        objective_function_values = Sf(x1, x2, x3, x4);
        objective_function_values = objective_function_values(:);

        out1 = x1(:)+1; out2 = x2(:)+1; out3 = x3(:)+1; out4 = x4(:)+1; % variables are normalized and centered around 1

    otherwise
        % Here is where you would load the actual databases from
end

    function r = rv(n) % random_vector
        r = normrnd(0,.05,[1 n]);
    end
    function z = zv(n) % zero vactor
        z = zeros([1 n]);
    end
end