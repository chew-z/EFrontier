function [sharpe, Sigma, mu] = ef(Mu, Cov, N, rf)
% returns effective frontier    
    if nargin < 3, N = 20;          end
    if nargin < 4, rf = 0.00001;    end

    mu = linspace(min(Mu), max(Mu), N);
    % Sigma = ones(1, N);
	f = zeros(length(Mu),1);
    Aeq = [ones(1,length(Mu)); Mu];
	w = zeros(length(Mu), N);
	options = optimset('Algorithm','interior-point-convex','TolFun',1e-10, 'Display', 'off');
    
    for i=1:N
        [x,~,fval] = quadprog(Cov,f,[],[],Aeq, [1; mu(i)],[],[],[],options);
		if fval > -1, 
            w(:,i) = x; 
            Sigma(i) = sqrt(w(:,i)'*Cov*w(:,i));
        else
            Sigma(i) = Inf;    
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