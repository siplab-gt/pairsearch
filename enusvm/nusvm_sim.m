function [] = nusvm_sim(dim,iterations,ntrials,noise_model,noise_norm,stagelist)
    % dim: embedding dimension
    % iterations: number of queries
    % ntrials: number of trials
    % noise_model: noise model (BT or NORMAL)
    % noise_norm: noise normalization
    % stagelist: [num_stage1,num_stage2,...]
    
    cvx_setup
    
    data_folder = './output-data';
    name = ['GaussCloud_d',num2str(dim),'_',num2str(iterations),'_',num2str(ntrials),'_',...
        noise_model,'_',noise_norm,'_',num2str(randi(10000))];
    
    fprintf(['dimension: %d\niterations: %d\nntrials: %d\n',...
        'noise_model: %s\nnoise normalization: %s\nstagelist: '],...
        dim,iterations,ntrials,noise_model,noise_norm);
    disp(stagelist)
    
    disp('loading embedding')
    [Y,kopt] = load_embedding(dim);
    klist_noise = [kopt.(noise_model).(noise_norm).('kopt')];
    
    p = size(Y,2); % dimension  
    nitems = size(Y,1);
    %if p > 4
    %    subsample_idx = randsample(nitems,floor(nitems/2));
    %    Y = Y(subsample_idx,:);
    %end
    %nitems = size(Y,1);
    
    R = sqrt(p);
    R_enforce = struct('value', R, 'enforce', 4, 'gamma', 0.7);
    maxiter = 600;
    
    err = nan(ntrials, length(stagelist), iterations+1);
    reach = nan(ntrials, length(stagelist), iterations, 2);
    
    W_hist = nan(ntrials, length(stagelist), iterations+1, p);
    W_sim = nan(ntrials, length(stagelist), p);
    
    %{
    defcolors = [
        0         0.4470    0.7410;
        0.8500    0.3250    0.0980;
        0.9290    0.6940    0.1250;
        0.4940    0.1840    0.5560;
        0.4660    0.6740    0.1880;
        0.3010    0.7450    0.9330;
        0.6350    0.0780    0.1840 ];
    %}
    
    sig = max(sqrt(diag(cov(Y)))); %0.7; % 0.5; % 0.3; % ;
    %disp('delaunayn');
    %T = delaunayn(Y);
    
    for trial = 1:ntrials
        %     Y = uniform_points_ball(nitems, p, R);
        %     x = randn(p, 1);
        %     x = xnorm * x / norm(x);
        
        for ii = 1:length(stagelist)
            x = rand(p,1)*2-1;
            W_sim(trial,ii,:) = x;
            
            nstages = stagelist(ii);
            
            measperstage = ones(nstages, 1) * floor(iterations / nstages);
            measperstage(end) = measperstage(end) ...
                + iterations - sum(measperstage);
            
            xhat = zeros(p, 1);
            
            W_hist(trial,ii,1,:) = xhat;
            err(trial, ii, 1) = norm(x - xhat);
            
            Ap = []; bp = []; yp = [];
            for jj = 1:nstages
                fprintf('trial %d / %d nstages %d / %d, stage %d / %d\n',trial,ntrials,ii,length(stagelist),...
                    jj,nstages);
                npairskeep = measperstage(jj);
                npairsgen = ceil(3*npairskeep);
                
                % mark's "closest point" method
                sigstage = sig ./ 2^(jj-1);
                Z = sigstage*randn(2*npairsgen, p) ...
                    + repmat(xhat', 2*npairsgen, 1);
                
                if jj > 1
                    0;
                end
                
                [ki, d] = dsearchn_alt2(Y,Z);
                %[ki, d] = dsearchn(Y, T, Z);
                
                aI = ki(1:npairsgen);
                bI = ki(npairsgen+1:end);
                
                bad = (aI == bI);
                aI(bad) = [];
                bI(bad) = [];
                
                d = reshape(d, npairsgen, 2);
                d(bad, :) = [];
                
                [~, ia] = unique([aI bI], 'rows');
                aI = aI(ia);
                bI = bI(ia);
                d = d(ia, :);
                
                if length(aI) > npairskeep
                    aI = aI(1:npairskeep);
                    bI = bI(1:npairskeep);
                    d = d(1:npairskeep, :);
                end
                npairsavail = length(aI);
                
                
                if length(aI) < npairskeep
                    error('not enough pairs generated');
                end
                
                
                A = ( Y(aI, :) - Y(bI, :) );
                b = ( sum(Y(aI, :).^2,2) - sum(Y(bI, :).^2,2) ) / 2;
                
                y = sign(A*x-b);
                
                A_alt = 2*A;
                b_alt = 2*b;
                
                num_err = 0;
                for kk = 1:length(y)
                    % errors
                    kval = klist_noise(1);
                    switch noise_norm
                        case 'CONSTANT'
                            z = A_alt(kk,:)*x-b_alt(kk,:);
                        case 'NORMALIZED'
                            z = (A_alt(kk,:)*x-b_alt(kk,:)) / norm(A_alt(kk,:));
                        case 'DECAYING'
                            z = (A_alt(kk,:)*x-b_alt(kk,:))*exp(-norm(A_alt(kk,:)));
                        otherwise
                            error('enter valid k normalization');
                    end
                    
                    %if strcmp(k_normalization, 'NONE')
                    %    z = 2*(A(kk,:)*x-b(kk,:));
                    %else
                    %    z = (A(kk,:)*x-b(kk,:)) / norm(A(kk,:));
                    %end
                    
                    switch noise_model
                        case 'BT'
                            prob_toggle = 1/(1+exp(kval*abs(z)));
                        case 'NORMAL'
                            prob_a = normcdf(z,0,1/kval);
                            prob_b = 1 - prob_a;
                            prob_toggle = (y(kk) == 1)*prob_b + (y(kk) < 1)*prob_a;
                        otherwise
                            error('enter valid noise model');
                    end
                    
                    if rand() < prob_toggle
                        y(kk) = -y(kk);
                        num_err = num_err + 1;
                    end
                    
                    if y(kk) < 0
                        y(kk) = -y(kk);
                        A(kk,:) = -A(kk,:);
                        b(kk) = -b(kk);
                    end
                end
                
                % CHEAT
                nu = 0.5;
                %nu = 2*num_err/numel(y);
                %fprintf('nu = %f\n',nu);
                
                x0 = xhat;
                for kk = 1:npairsavail   % npairskeep
                    %xhat = solveprob([Ap; A(1:kk,:)]', [bp; b(1:kk)], x0);
                    xhat = enusvm_solve([Ap; A(1:kk,:)]', [bp; b(1:kk)], ...
                        [yp; y(1:kk)], nu, maxiter, R_enforce);
                    measplace = kk;
                    if jj > 1
                        measplace = measplace + sum(measperstage(1:jj-1));
                    end
                    
                    xhat = min(max(xhat,-1),1); % project to unit cube
                    W_hist(trial,ii,measplace+1,:) = xhat;
                    
                    err(trial, ii, measplace+1) = norm(x - xhat);
                    reach(trial, ii, measplace, 1) = d(kk, 1);
                    reach(trial, ii, measplace, 2) = d(kk, 2);
                end
                
                if 1   % keep measurements
                    Ap = [Ap; A];
                    bp = [bp; b];
                    yp = [yp; y];
                end
            end
        end
    end
    
    save([data_folder,'/',name,'.mat'],'W_hist','W_sim','dim','iterations',...
    'ntrials','noise_model','noise_norm','stagelist');
end

function [k,d] = dsearchn_alt(X,Xi)
    ni = size(Xi,1);
    k = zeros(ni,1);
    d = zeros(ni,1);
    
    for ii=1:ni
        xi = Xi(ii,:);
        dist = sqrt(sum((X-xi).^2,2));
        [d(ii),k(ii)] = min(dist);
    end
    
end

function [k,d] = dsearchn_alt2(X,Xi)
    ni = size(Xi,1);
    assert(mod(ni,2)==0);
    
    k = zeros(ni,1);
    d = zeros(ni,1);
    
    for ii=1:ni/2
        xi = Xi(ii,:);
        xi2 = Xi(ii+ni/2,:);
        dist = sqrt(sum((X-xi).^2,2));
        dist2 = sqrt(sum((X-xi2).^2,2));
        [d(ii),k(ii)] = min(dist);
        
        % enforce unique pairs
        dist2(k(ii)) = inf;
        
        kii_idx = find(k == k(ii));
        kii_idx = kii_idx(:);
        pair_idx = [(kii_idx(kii_idx <= ni/2) + ni/2);(kii_idx(kii_idx > ni/2) - ni/2)];
        k_pair = k(pair_idx);
        k_pair = unique(k_pair(k_pair > 0));
        
        dist2(k_pair) = inf;
        [d(ii+ni/2),k(ii+ni/2)] = min(dist2);
    end
    
end
