function [sharpe, Sigma, mu] = ef(Mu, Cov, bd, N, rf)
% searches for returns effective frontier in two steps
% more smooth around tangency (max(sharpe))
% bd - flag [0, 1]. if bd == 1 Sum(w(i)) = 1    
    if nargin < 4, N = 20;          end
    if nargin < 5, rf = 0.00001;    end

    mu = linspace(min(Mu), 5*max(Mu), N);   % How one can find good range (not too wide, not too narrow)? 
    % The equally spaced search here is ineffective. We should look in a range [mu(min(sigma)) mu(max(sharpe))] with points concentrated
    % around mu(max(sharpe)) - tangency portfolio.
    % Sigma = ones(1, N);

    f = zeros(length(Mu),1);
    if bd == 1
        Aeq = [ones(1,length(Mu))];
        beq = [1];
    else
        Aeq = [];
        beq = [];
    end 
    options = optimset('Algorithm','interior-point-convex','TolFun',1e-10, 'Display', 'off');
    for i=1:N
        [x,~,fval] = quadprog(Cov,f,[],[],[Aeq; Mu], [beq; mu(i)],[],[],[],options);
		if fval > -1, 
            Sigma(i) = sqrt(x'*Cov*x);
        else
            Sigma(i) = Inf;    % There could be many reasons quadprog returns negative fval. Zero (iterations exceeded maxIter).
                               % is also not such a great result. Check input. Run with 'Display', 'iter'. 
        end 
    end
    sharpe = (mu-rf) ./ Sigma;
    imin = find(Sigma == min(Sigma));
    imax = find(sharpe == max(sharpe));
    % if imin <> imax
    mu = [linspace(mu(imin), mu(imax-1), N), linspace(mu(imax-1), mu(imax+1), N)];  % overlap in mu(imax-1)
    w = zeros(length(Mu), 2*N);
    Sigma = ones(1, 2*N);
    % Now repeat
    for i=1: 2*N
        [x,~,fval] = quadprog(Cov,f,[],[],[Aeq; Mu], [beq; mu(i)],[],[],[],options);
        if fval > -1, 
            w(:,i) = x; 
            Sigma(i) = sqrt(w(:,i)'*Cov*w(:,i));
        else
            Sigma(i) = Inf;    % There could be many reasons quadprog returns negative fval. Zero (iterations exceeded maxIter).
                               % is also not such a great result. Check input. Run with 'Display', 'iter'. 
        end 
    end
    sharpe = (mu-rf) ./ Sigma;

    % sigma_eff = Sigma((mu >= mu(imin)));
    % m_eff = mu(mu > mu(imin));
    % sh = sharpe(mu > mu(imin));
    % i_tang = find(sharpe == max(sharpe));
    % weight_tang = w(:,i_tang);
    % sharpe_tang = sharpe(i_tang);

end	