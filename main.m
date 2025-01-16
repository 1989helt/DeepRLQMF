clear
warning off

q2double = @(X) double2q(X,'inverse');
GT = double(imread('house.png'));  
Y = double2q(GT);
[n1, n2, n3] = size(GT);
M = zeros(n1, n2);
rate = 0.25; %MR value
omega = rand(n1 * n2 , 1) < rate; %mask 
M(omega) = 1;
omega1 = find(M); %omega index
%%%opts
opts.lambda1 = 5 * sqrt(max(n1, n2));
opts.lambda2 = 0.1;
opts.varepsilon = 1e-8;
opts.c = 1;
opts.tol = 0.001;
opts.rank = 10;
opts.p = 0.1;
opts.numIter = 300;
%%%
imgN = double2q(zeros(n1,n2,n3));
imgN(omega1) = Y(omega1); %dirty image
psnr1 = psnr(double(GT)./255, q2double(imgN)./255);%dirty image psnr value
ssim1 = ssim(double(GT)./255, q2double(imgN)./255);%dirty image ssim value
fprintf('PSNR is %2.2f dB\n', psnr1);
fprintf('SSIM is %2.4f dB\n', ssim1);
figure;
%PnP
% [U,V] = DQRLMFPnP(imgN,M,opts,50);
%RED
% [U,V] = DQRLMFRED(imgN,M,opts,15);
%QRLMF
[U,V] = QRLMF(imgN,M,opts);
%Output X
X = U * V';
X(omega1)  =imgN(omega1);
X = q2double(X); 
psnr2 = psnr(X./255,double(GT)./255 );
SSIM = ssim(double(GT)./255, X./255);
fprintf('PSNR achieved by QRLMF is %2.2f dB\n', psnr2);
fprintf('SSIM achieved by QRLMF is %2.4f dB\n', SSIM);

