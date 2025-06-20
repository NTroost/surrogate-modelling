function [out1,out2,out3,out4,errors_profile, d, PUhn, UTS, sy, F, deltaD] = getData(db, num)
% Sample data points (X, y, z, q) and corresponding objective function values S

switch 1
    case strcmp(db, 'test')
        n = 500; % amount of discrete points
        yms = normrnd(0,.05,[1 n]); % Replace with your actual data
        s0s = normrnd(0,.05,[1 n]); % Replace with your actual data
        sys = normrnd(0,.05,[1 n]); % Replace with your actual data
        nexps = normrnd(0,0.05,[1 n]); % Replace with your actual data

        Hd = [10 50 100 500];
        if num == 4
            Sf = @(s1, s2, s3, s4) ...
                0.5*Hd(1)*s1.^2 + 0.5*Hd(2)*s2.^2 + 0.5*Hd(3)*s3.^2 + 0.5*Hd(4)*s4.^2 + ...
                sqrt(Hd(1)*Hd(2))*s1.*s2 + ...
                sqrt(Hd(1)*Hd(3))*s1.*s3 + ...
                sqrt(Hd(1)*Hd(4))*s1.*s4 + ...
                sqrt(Hd(2)*Hd(3))*s2.*s3 + ...
                sqrt(Hd(2)*Hd(4))*s2.*s4 + ...
                sqrt(Hd(3)*Hd(4))*s3.*s4; % sqrt(Hs1*Hs2)
            errors_profile = Sf(yms, s0s, sys, nexps);
        elseif num == 3
            Sf = @(s1, s2, s3) ...
                0.5*Hd(1)*s1.^2 + 0.5*Hd(2)*s2.^2 + 0.5*Hd(3)*s3.^2 + ...
                sqrt(Hd(1)*Hd(2))*s1.*s2 + ...
                sqrt(Hd(1)*Hd(3))*s1.*s3 + ...
                sqrt(Hd(2)*Hd(3))*s2.*s3; % sqrt(Hs1*Hs2)
            errors_profile = Sf(yms, sys, nexps);
        end

        out1 = yms+1; out2 = s0s+1; out3 = sys+1; out4 = nexps+1;
        d = NaN;
        PUhn = NaN;
        UTS = [];
        sy = [];
        F = [];
        deltaD = [];

    otherwise

        %% Init
        % base_direc = 'C:\Users\nchtroost\OneDrive - Delft University of Technology\Documents\Documents\RESURGAM\31_Testing\31_H2S2\ANSYS\Indenter\';
        base_direc = 'E:\RESURGAM\Databases storage\IFEM';
        sub_direc = db;
        n = 1000;
        PU_only = false;

        %% read
        direc = [base_direc  '\' sub_direc];
        subfolders = dir(direc);

        %% read base ld curve
        % base_ld_folder1 = '640000-200000_0-499_5-500_0-0_2-0_3-1';
        % base_ld_folder2 = '640000-200000_0-300_0-500_0-0_2-0_3-1';
        base_ld_folder1 = '210000-122500_0-971_7-1151-0_0386-0_34-1';
        base_ld_folder2 = '210000-80230_0-251_9-1124-0_4033-0_35-1';
        base_ld_folder3 = '210000-70810_0-282_4-397_9-0_1181-0_33-1';
        base_ld_folder4 = '210000-215100_0-378_7-769_3-0_1446-0_3-1';
        base_ld_folder5 = '210000-122500_0-920_2-17_87-1-0_34-1';
        base_ld_folder6 = '210000-80230_0-106_4-2_657-1-0_35-1';
        base_ld_folder7 = '210000-70810_0-249_2-11_94-1-0_33-1';
        base_ld_folder8 = '210000-215100_0-287_6-5_549-1-0_3-1';
        base_ld_folder9 = '210000-110860_0-726_17-1500-0_51356-0_3-1';  % dummy
        if isfile(fullfile(direc, base_ld_folder1, '10_PROFILE.txt'))
            base_ld_folder = base_ld_folder1;
        elseif isfile(fullfile(direc, base_ld_folder2, '10_PROFILE.txt'))
            base_ld_folder = base_ld_folder2;
        elseif isfile(fullfile(direc, base_ld_folder3, '10_PROFILE.txt'))
            base_ld_folder = base_ld_folder3;
        elseif isfile(fullfile(direc, base_ld_folder4, '10_PROFILE.txt'))
            base_ld_folder = base_ld_folder4;
        elseif isfile(fullfile(direc, base_ld_folder5, '10_PROFILE.txt'))
            base_ld_folder = base_ld_folder5;
        elseif isfile(fullfile(direc, base_ld_folder6, '10_PROFILE.txt'))
            base_ld_folder = base_ld_folder6;
        elseif isfile(fullfile(direc, base_ld_folder7, '10_PROFILE.txt'))
            base_ld_folder = base_ld_folder7;
        elseif isfile(fullfile(direc, base_ld_folder8, '10_PROFILE.txt'))
            base_ld_folder = base_ld_folder8;
        elseif isfile(fullfile(direc, base_ld_folder9, '10_PROFILE.txt'))
            base_ld_folder = base_ld_folder9;
        else
            warning('No base folder')
            keyboard
        end
        disp('Base folder identified')
        pname = fullfile(direc, base_ld_folder, '10_PROFILE.txt');
        [profile, material_properties] = read_db(pname);
        profile = profile * n; [xq,ind] = sort(profile(:,1));
        norm = sqrt(trapz(xq, profile(ind,2).^2) / 1e3);

        % interpolate
        ax = profile(:,1); [ax, I] = sort(ax);
        ay = interp1(profile(I,1), profile(I,2), ax);
        profile = [ax(:) ay(:)];

        inds = profile(:,2) < 0;
        if PU_only
            % inds = profile(:,1) < 0.5*1000;
            profile(inds,2) = 0;
            profile(inds,2) = min(profile(:,2));
        else
            dfac = 1/(0.4 / (-min(profile(:,2))) * 200);
            profile(inds,2) = profile(inds,2) / dfac;
        end

        % Correct for neutral line
        % relevant in actual

        %% init vectors
        yms = []; s0s = []; sys = []; nexps = [];

        errors_profile = [];
        d = [];
        PUhn = [];
        UTS = [];
        sy = [];
        F = [];
        deltaD = [];

        for subf_idx = 1:length(subfolders)

            %% Folder filter
            fname = fullfile(direc, subfolders(subf_idx).name, '10_MICRO.txt');
            pname = fullfile(direc, subfolders(subf_idx).name, '10_PROFILE.txt');
            mainname = fullfile(direc, subfolders(subf_idx).name, 'main.ans');

            % mat = material_properties_from_filename(fname);
            %
            % try
            %     figure(1); plot(mat(2), mat(3), '.'); hold on
            %     figure(2); plot(mat(2), mat(4), '.'); hold on
            %     figure(3); plot(mat(2), mat(5), '.'); hold on
            %     figure(4); plot(mat(3), mat(4), '.'); hold on
            %     figure(5); plot(mat(3), mat(5), '.'); hold on
            %     figure(6); plot(mat(4), mat(5), '.'); hold on
            % catch
            % end

            if strcmp(subfolders(subf_idx).name, base_ld_folder)
                % continue
            end

            %% Read
            if isfile(pname)
                try
                    [x, mat_props] = read_db(pname);
                    x = x*n;
                catch
                    keyboard
                end
            elseif ~exist(mainname, 'file')
                continue
            else
                disp(subfolders(subf_idx).name)
                continue
            end
            if isfile(fname)
                ld = read_db(fname);
                uidx = find(ld(:,2) < max(ld(:,2))/1000, 1, 'first');
                if ~isempty(uidx)
                    deltaD(end+1) = 2 * (-min(x(:,2))/1000 - ld(uidx));
                else
                    deltaD(end+1) = 2 * (-min(x(:,2))/1000 - ld(end));
                end
                F(end+1) = max(ld(:,2));


            else
                F(end+1) = -1;
                deltaD(end+1) = -2;
            end

            % sort and interpolate
            [axs, I] = sort(x(:,1)); x = [axs(:) x(I,2)];
            ay = interp1(x(:,1), x(:,2), ax, 'spline', 'extrap');
            x = [ax(:) ay(:)];

            %%
            sy(end+1) = mat_props(4);
            if mat_props(5) == 1
                UTS(end+1) = estimate_UTS_eng_RO(mat_props(2), mat_props(3), mat_props(4));
            else
                UTS(end+1) = estimate_UTS_voce(mat_props(2), mat_props(3), mat_props(4), mat_props(5));
                if UTS(end) > mat_props(4)
                    UTS(end) = mat_props(3);
                end
            end

            %%
            rq = linspace(min(x(:,1)), max(x(:,1)), 1001);
            zq = interp1(x(:,1),x(:,2),rq,"pchip");
            gx = -gradient(gradient(zq) ./ gradient(rq));
            idxg = find(gx > max(gx)*0.9, 1, 'first');
            d1 = 2 * rq(idxg);

            idx = [];
            h_target = 0;
            count = 0;
            while isempty(idx)
                idx = find(x(:,2) >= h_target, 1, 'first');
                h_target = h_target + min(x(:,2))/100;
                count = count+1;
            end
            x1 = x(idx-1,1);
            x2 = x(idx,1);
            y1 = x(idx-1,2);
            y2 = x(idx,2);
            d2 = 2 * (x1 - y1 * (x2 - x1) / (y2 - y1));

            d(end+1) = d2;
            PUhn(end+1) = zq(idxg) / - min(x(:,2));

            % if d2 < d1
            %     PUhn(end+1) = max(x(:,2)) / - min(x(:,2));
            % else
            %     PUhn(end+1) = x(idxg,2) / - min(x(:,2));
            % end
            % [~,idx] = max(x(:,2));
            % d(end) = 2 * x(idx,1);
            
            %%
            inds = x(:,2) < 0;
            if PU_only
                % inds = x(:,1) < 0.5*1000;
                x(inds,2) = 0;
                x(inds,2) = min(x(:,2));
            else
                x(inds,2) = x(inds,2) / dfac;
            end

            %%
            ym = mat_props(2); yms = [yms ym/material_properties(2)];
            s0 = mat_props(3); s0s = [s0s s0/material_properties(3)];
            sy = mat_props(4); sys = [sys sy/material_properties(4)];
            nexp = mat_props(5); nexps = [nexps nexp/material_properties(5)];

            %% compare profile
            error = sqrt(trapz(ax(:), (x(:,2) - profile(:,2)).^2)) / norm;
            errors_profile = [errors_profile error];

            %% Old code
            % [xq, si] = unique(profile(:,1));
            % profile = [xq profile(si,2)];
            %
            % profile(:,2) = profile(:,2)-min(profile(:,2)) + min(base_case{2}(:,2));
            %
            % % pile up only
            % pu = profile; ed = profile;
            % pu(profile(:,1) < zerocrossing, 2) = 0;
            % ed(profile(:,1) > zerocrossing_down, 2) = 0;
            %
            % % correct measured profile because neutral line is unknown
            % fun = @(nl) trapz(bxq, ((bpu(:,2) - nl*max(abs(base_case{2}(:,2)))) - ...
            %     interp1(pu(:,1),pu(:,2),bxq,'linear','extrap')).^2);
            % nl = 0*fminsearch(fun, 0) * max(abs(base_case{2}(:,2)));
            %
            % % compute error
            % ep = trapz(bxq, ((base_case{2}(:,2) - nl) - ...
            %     interp1(profile(:,1),profile(:,2),bxq,'linear','extrap')).^2);
            % epu = trapz(bxq, ((bpu(:,2) - nl) - ...
            %     interp1(pu(:,1),pu(:,2),bxq,'linear','extrap')).^2);
            % eed = trapz(bxq, ((bed(:,2) - nl) - ...
            %     interp1(ed(:,1),ed(:,2),bxq,'linear','extrap')).^2);
            %
            % errors_profile = [errors_profile ep];
            % errors_pu = [errors_pu epu];
            % errors_ed = [errors_ed eed];
        end

        out1 = yms;
        out2 = s0s;
        out3 = sys;
        out4 = nexps;
end

function UTS_eng = estimate_UTS_voce(E, sigma_0, sigma_s, epsilon_0)

    % Define Voce true stress function
    sigma_true = @(eps) sigma_s - (sigma_s - sigma_0) .* exp(-eps ./ epsilon_0);

    % Define derivative of true stress
    dsigma_deps = @(eps) (sigma_s - sigma_0) ./ epsilon_0 .* exp((sigma_0 / E - eps) ./ epsilon_0);

    % Necking condition: dσ_true/dε = σ_true
    necking_eq = @(eps) dsigma_deps(eps) - sigma_true(eps);

    % Solve for strain at necking (true strain)
    eps_neck = fzero(necking_eq, 0.1);  % Initial guess
    if eps_neck < 0
        eps_neck = 0;
    end

    % Calculate engineering UTS
    sigma_true_neck = sigma_true(eps_neck);
    UTS_eng = sigma_true_neck / (1 + eps_neck);

    eps = linspace(0,0.2,1001);
end
function UTS_eng = estimate_UTS_eng_RO(E, Rp02, n)

    % Estimates engineering UTS from Ramberg-Osgood parameters
    % Inputs:
    %   E     - Young's modulus [MPa]
    %   Rp02  - Yield strength [MPa]
    %   n     - Strain hardening exponent [-]
    % Output:
    %   UTS_eng - Engineering UTS [MPa]

    % Create stress range: from Rp02 to ~10×Rp02
    sigma = linspace(Rp02, Rp02 * 10, 10000);
    
    % True strain from Ramberg-Osgood relation
    eps_true = sigma / E + 0.002 * (sigma / Rp02).^n;

    % Numerical derivative dσ/dε
    dsig = gradient(sigma);
    deps = gradient(eps_true);
    d_sigma_d_eps = dsig ./ deps;

    % Criterion: find where dsig/de = sig/e
    % Compute error between two sides
    error_ = abs(d_sigma_d_eps - sigma);
    [~, idx_] = min(error_);  % Find index of minimum error

    % Get true stress and true strain at necking
    sigma_uts_true = sigma(idx_);
    eps_uts_true = eps_true(idx_);

    % Convert to engineering UTS
    UTS_eng = sigma_uts_true / (1 + eps_uts_true);
end

end