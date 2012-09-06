% finds efficient frontiers t(k)..t(k+N) and plots results
% input - covariance matrices and mean returns vectors

K = 25;			% how many points per each ef
N = 10;			% how many steps
Offset = 200;	% where to start
sharpe= zeros(N, K); sigma = zeros(N, K); mu = zeros(N, K);
bounded = 1; % 1 - Sum(weights) = 1

msgid = 'optim:quadprog:HessianNotSym'; % Quadprog checks norm(H-H',inf) > eps 
% Ignore this warning, my matrices are almost symetric to the precision (e-25 vs. e-18)
warning('off', msgid);

for j=Offset+1:Offset+N
	m = M(j,:); cv = CV(:,:,j);
	[sharpe(j-Offset,:), sigma(j-Offset,:), mu(j-Offset,:)] = ef(m, cv, bounded, K);
end
warning('on', msgid);

% plot
figure(1); clf(1);
subplot(2,1,1); 		title('Sharpe vs sigma'); xlabel('sigma'); ylabel('sharpe');
for j = 1:N 
	hold all; plot(sigma(j,:), sharpe(j,:));
end
subplot(2,1,2); 		title('Efficient frontier'); xlabel('sigma'); ylabel('returns');
for j = 1:N
	hold all; plot(sigma(j,:), mu(j,:));
end	
hold off

clear m cv j Offset K N msgid