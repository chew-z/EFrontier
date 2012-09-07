function [m_tang, sigma_tang, sharpe_tang, weight_tang] = tangency(Mu, Cov, N, rf, w0)
% returns tangency portfolio    
    if nargin < 3, N = 20;          end
    if nargin < 4, rf = 0.00001;    end
    if nargin < 5, w0 = [];         end
	mu = linspace(min(Mu), max(Mu), N);
	
	f      = zeros(length(Mu),1);
    Aeq    = [ones(1,length(Mu)); Mu];
	w      = zeros(length(Mu), N);
 	options = optimset('Algorithm','interior-point-convex','TolFun',1e-10, 'Display', 'off');
    for i=1:N
		[x,~,fval] = quadprog(Cov,f,[],[],Aeq, [1; mu(i)],[],[],w0,options);
		if fval > -1, 
            w(:,i) = x; 
            sigma(i) = sqrt(w(:,i)'*Cov*w(:,i));
        end    
    end
    sharpe = (mu-rf) ./ sigma;
    i_tang = find(sharpe == max(sharpe));
    it = i_tang(1);             % if length(i_tang) > 1 --> warning etc...
    weight_tang = w(:,it);
    sharpe_tang = sharpe(it); % == max(sharpe)
    m_tang = mu(it); sigma_tang = sigma(it);
end	