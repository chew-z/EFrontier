function [sharpe, Sigma, mu] = ef(Mu, Cov, bd, N, rf)
% returns effective frontier
% bd - flag [0, 1]. if bd == 1 Sum(w(i)) = 1    
    if nargin < 4, N = 20;          end
    if nargin < 5, rf = 0.00001;    end

    mu = linspace(min(Mu), max(Mu), N);
    % Sigma = ones(1, N);
	f = zeros(length(Mu),1);
    if bd == 1
        Aeq = [ones(1,length(Mu))];
        beq = [1];
    else
        Aeq = [];
        beq = [];
    end            
	w = zeros(length(Mu), N);
	options = optimset('Algorithm','interior-point-convex','TolFun',1e-10, 'Display', 'iter');
    
    for i=1:N
        [x,~,fval] = quadprog(Cov,f,[],[],[Aeq; Mu], [beq; mu(i)],[],[],[],options);
		if fval > -1, 
            w(:,i) = x; 
            Sigma(i) = sqrt(w(:,i)'*Cov*w(:,i));
        else
            Sigma(i) = Inf;    % There could be many reasons qudprog returns negative fval. Zero (iterations exceeded maxIter).
                               % is also not such a great result. Check input. Run with 'Display', 'iter'. 
        end 
    end
    sharpe = (mu-rf) ./ Sigma;
    imin = find(Sigma == min(Sigma));

    sigma_eff = Sigma((mu >= mu(imin)));
    m_eff = mu(mu > mu(imin));
    sh = sharpe(mu > mu(imin));
    % i_tang = find(sharpe == max(sharpe));
    % weight_tang = w(:,i_tang);
    % sharpe_tang = sharpe(i_tang);

end	