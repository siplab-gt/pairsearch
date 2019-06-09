function [x_hat, xi, cinfo] = enusvm_solve(A, b, y, nu, maxiter, mopts)

if nargin < 6 || isempty(mopts)
    normconstraint = 0;
    mopts = [];
else
    normconstraint = mopts.enforce;  % 0, 1, 2, 3, or 4
    R = mopts.value;
end

[n, p] = size(A);

% get the "admissible" range of nu
examp = [A' -b];
bias = 0;  % set to 1 to have bias term optimized

C = 1e6;
cvx_begin quiet
    cvx_precision default
    variable w_hat(n+1)
    if bias > 0, variable bias, end
    variable xi(p)
    %
    dual variable d_alpha
    dual variable d_beta
    %
    minimize 0.5*sum_square(w_hat) + C*sum(xi)
    subject to
        d_alpha : y.*(examp*w_hat + bias) >= 1 - xi;         %#ok
        d_beta  : xi >= 0;                                   %#ok
cvx_end

nu_min = sum(d_alpha) / (C*p);

% 1: Enusvm, 0: single-pass
extended = 1;

if nu > nu_min
    extended = 0;
    nu0 = nu;
else
    nu0 = 1.01*nu_min;
end

% starting point
cvx_begin quiet
    cvx_precision default
    variable w_tilde(n+1)
    if bias > 0, variable bias, end
    variable xi(p)
    variable rho
    %
    dual variable d_alpha
    dual variable d_beta
    dual variable d_delta
    %
    minimize 0.5*sum_square(w_tilde) - nu0*rho + (1/p)*sum(xi)
    subject to
        d_alpha : y.*(examp*w_tilde + bias) >= rho - xi;        %#ok
        d_beta  : xi >= 0;                                      %#ok
        d_delta : rho >= 0;                                     %#ok
cvx_end

random_init = 0;
if isnan(d_delta) || any(isnan(w_tilde))
    fprintf('bad starting point: status %s, nu %g, min %g, ext %d\n', ...
        cvx_status, nu0, nu_min, extended);
    random_init = 1;
end

if ~extended
    fprintf('ordinary nu-svm success\n');
    x_hat = w_tilde(1:n)/w_tilde(n+1);

    cinfo.iters = 0;
    cinfo.gap = 0;
    cinfo.w_hat = [];
    cinfo.rho = rho;
    cinfo.nu_min = nu_min;

    return
end

if random_init
    w_tilde = randn(n+1,1) / sqrt(n+1);
end

% Enu svm
t = tic;
tol = 1e-3;

gam = 9/10;

if isfield(mopts, 'gamma')
    gam = mopts.gamma;
end
    
if nargin < 5
    maxiter = 100;
end
for ii = 1:maxiter

    % no bias
    % -y.*examp*w_hat + rho - xi <= 0
    % -xi <= 0
    % w_tilde'*w_hat == 2

    lpf = [ -nu, ones(1,p)/p, zeros(1,n+1) ];
    lpA = [ ones(p,1),     -eye(p),      diag(-y)*examp; ...
            zeros(p,1),    -eye(p),      zeros(p,n+1)  ];

    lpb = zeros(2*p,1);

    if normconstraint == 1
        constraint2 = [ w_tilde(1:n); 0 ] ./ w_tilde(n+1)^2;
        lpA = [lpA; [zeros(1,p+1), constraint2']];
        lpb = [lpb; R];
    end

    lpAeq = [ zeros(1,p+1), w_tilde' ];
    opts = optimoptions('linprog', 'Display', 'off', ...
        'Algorithm', 'dual-simplex');
    lp = linprog(lpf, lpA, lpb, lpAeq, 2, [], [], opts);
    w_hat = lp(2+p:end);
    xi = lp(2:p+1);

%{
    cvx_begin quiet
        cvx_precision default
        variable w_hat(n+1)
        if bias > 0, variable bias, end
        variable xi(p)
        variable rho
        %
        dual variable d_alpha
        dual variable d_beta
        %
        minimize -nu*rho + (1/p)*sum(xi)
        subject to
            d_alpha : y.*(examp*w_hat + bias) >= rho - xi;       %#ok
            d_beta  : xi >= 0;                                   %#ok
            w_tilde'*w_hat == 2;                                 %#ok
    cvx_end
%}

    if norm(w_tilde - w_hat) <= tol*norm(w_hat)
        break;
    end

    if normconstraint == 2 || normconstraint == 3
        x_hat_ = w_hat(1:n)./w_hat(n+1);
        if norm(x_hat_) > R
            x_hat_ = x_hat_ ./ norm(x_hat_) * R;
            w_hat = [x_hat_; 1];
            w_hat = w_hat / norm(w_hat);
        end
    end
    if normconstraint == 2
        x_tilde_ = w_tilde(1:n)./w_tilde(n+1);
        if norm(x_tilde_) > R
            x_tilde_ = x_tilde_ ./ norm(x_tilde_) * R;
            w_tilde = [x_tilde_; 1];
            w_tilde = w_tilde / norm(w_tilde);
        end
    end

    w_tilde = gam*w_tilde + (1-gam)*w_hat;

end

cinfo.iters = ii;
cinfo.gap = norm(w_tilde - w_hat)/norm(w_hat);
cinfo.w_hat = w_hat;
cinfo.rho = lp(1);
cinfo.nu_min = nu_min;

fprintf('iterations %d, gap %d, time %gs\n', ...
    ii, norm(w_tilde-w_hat)/norm(w_hat), toc(t));

x_hat = w_hat(1:n)/w_hat(n+1);

if normconstraint == 4
    if norm(x_hat) > R
        x_hat = x_hat ./ norm(x_hat) * R;
    end
end


