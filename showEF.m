% finds efficient frontiers t(k)..t(k+N) and plots results
% input - covariance matrices and mean returns vectors
% plots efficient frontier, marks minimum variance point(o) and max sharpe point(x)
% 

K = 10;			% how many points per each frontier
N = 1;			% how many time steps
Offset = 350;	% where to start
sharpe= zeros(N, 2*K); sigma = zeros(N, 2*K); mu = zeros(N, 2*K);
sharpe0= zeros(N, 2*K); sigma0 = zeros(N, 2*K); mu0 = zeros(N, 2*K);
sh = zeros(N,1); si = zeros(N,1); mi = zeros(N,1);
sht = zeros(N,1); sit = zeros(N,1); mit = zeros(N,1);
bounded = 1; % 1 - Sum(weights) = 1

msgid = 'optim:quadprog:HessianNotSym'; % Quadprog checks norm(H-H',inf) > eps 
% Ignore this warning, my matrices are almost symetric to the precision (e-25 vs. e-18)
warning('off', msgid);

for j=Offset+1:Offset+N
	m = M(j,:); cv = CV(:,:,j);
	[it, sharpe(j-Offset,:), sigma(j-Offset,:), mu(j-Offset,:)] = ef2(m, cv, 1, K);						% effective frontier
	[~, sharpe0(j-Offset,:), sigma0(j-Offset,:), mu0(j-Offset,:)] = ef2(m, cv, 0, K);							% combination of tangency portfolio and risk-free asset allocation
	[sh(j-Offset), si(j-Offset), mi(j-Offset)] = minvar(m, cv);													% minimum variance point
	if length(it) > 1, it = it(1); end	% give warning
	sht(j-Offset) = sharpe(j-Offset,it); sit(j-Offset) = sigma(j-Offset,it); mit(j-Offset) = mu(j-Offset,it);	% max sharpe point
end
warning('on', msgid);

% plot
figure(1); clf(1);
subplot(2,1,1); 		title('Sharpe vs sigma'); xlabel('standard deviation of returns'); ylabel('sharpe ratio');
for j = 1:N 
	hold all; plot(sigma(j,:), sharpe(j,:));
	plot(sigma0(j,:), sharpe0(j,:)); 
	plot(si(j), sh(j),'o-','MarkerSize',10); 
	plot(sit(j), sht(j),'x-','MarkerSize',10);
end
subplot(2,1,2); 		title('Efficient frontier'); xlabel('standard deviation of returns'); ylabel('expected returns');
for j = 1:N
	hold all; plot(sigma(j,:), mu(j,:));
	plot(sigma0(j,:), mu0(j,:)); 
	plot(si(j), mi(j), 'o-','MarkerSize',10); 
	plot(sit(j), mit(j), 'x-','MarkerSize',10);
	line([0,sit(j)], [0, mit(j)]);
end

hold off

% clean the mess with variables
clear m cv j Offset K N msgid bounded