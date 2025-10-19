function [U,V] = DQRLMFPnP(Y,M,opts,alpha)
q2double = @(X) double2q(X,'inverse');
%%%init
lambda1 = opts.lambda1;
p = opts.p;
varepsilon = opts.varepsilon;
c = opts.c; 
tol = opts.tol;
[n1, n2] = size(Y);
omega = find(M);  
Yomega = Y(omega);
weight = @(x) c .* (x + varepsilon) .^ (p - 1); 
rank = opts.rank;%rank value
numIter = opts.numIter; 
%init U1
zerou1 = zeros(n1,rank);
U1i = randn(n1, rank);
U1j = randn(n1, rank);
U1k = randn(n1, rank);
U1 = quaternion(zerou1,U1i,U1j,U1k);
%init V1
zerou2 = zeros(n2,rank);
V1i = randn(n2,rank);
V1j = randn(n2,rank);
V1k = randn(n2,rank);
V1 = quaternion(zerou2,V1i,V1j,V1k);
V = V1;
%init X and Z
X = U1 * V1'; 
Z = zeros(n1,n2,3);
[uu1,s1,vv1] = qsvd(U1); 
W_u = diag(s1);
[uu2,s2,vv2] = qsvd(V1); 
W_v = diag(s2);
Z1 = quaternion(zeros(n1, rank),zeros(n1, rank),zeros(n1, rank),zeros(n1, rank));
Z2 = quaternion(zeros(n2,rank),zeros(n2,rank),zeros(n2,rank),zeros(n2,rank));
Z3 = quaternion(zeros(n1, n2),zeros(n1, n2),zeros(n1, n2),zeros(n1, n2));
Z4 = zeros(n1,n2,3);
mu = 0.001;
rho = 1.1;
%%%update iter
for iter = 1 : numIter
  %update U
  U = (U1 - Z1 ./mu + (X + Z3 ./ mu) * V) * inv(eye(rank) + V' * V);
  %update V
  V = (V1 - Z2 ./mu + (X + Z3 ./ mu)' * U) * inv(eye(rank) + U' * U)  ; 
  %update U1
  [a1, b1, c1] = qsvd(U + Z1 ./ mu);
  W_u = diag(b1) ./ (lambda1 / mu .* weight(W_u .* W_v) + 1);
  Qua_a1 = quaternion(a1(:,:,1),a1(:,:,2),a1(:,:,3),a1(:,:,4));
  Qua_c1 = quaternion(c1(:,:,1),c1(:,:,2),c1(:,:,3),c1(:,:,4));
  U1 = Qua_a1 * diag(W_u) * Qua_c1';
  %update V1
  [a2, b2, c2] = qsvd(V + Z2 ./ mu);
  W_v = diag(b2) ./ (lambda1 / mu .* weight(W_u .* W_v) + 1);
  Qua_a2 = quaternion(a2(:,:,1),a2(:,:,2),a2(:,:,3),a2(:,:,4));
  Qua_c2 = quaternion(c2(:,:,1),c2(:,:,2),c2(:,:,3),c2(:,:,4));
  V1 = Qua_a2 * diag(W_v) * Qua_c2';
  %update X
  QZ4 = double2q(Z4);
  temp = U * V';
  QZ = double2q(Z);
  X = (mu .* temp - Z3 + mu .* QZ - QZ4 )./(mu+mu);
  tempc = (1 + mu + mu);
  X(omega) = (mu .* temp(omega) - Z3(omega) + mu .* QZ(omega) - QZ4(omega) + Yomega) ./ tempc;
  %update Z-PnP
  QX = q2double(X);
  tempZ = QX + Z4./mu;
  Z = FFDNet(tempZ./255,sqrt(alpha./mu)./255).*255;
  %update Larangian  
  Z1 = Z1 + mu .* (U - U1); 
  Z2 = Z2 + mu .* (V - V1); 
  Z3 = Z3 + mu .* (X - temp);     
  Z4 = Z4 + mu .* ( QX - Z);   
  mu = mu * rho;
  %Error
  loss_U = (norm((U - U1), 'fro') / norm((U1), 'fro'))^2;
  loss_V = (norm((V - V1), 'fro') / norm((V1), 'fro'))^2;
  loss_UV = (norm((temp - X), 'fro') / norm((temp), 'fro'))^2;
  loss_Z = (norm(double2q(QX - Z), 'fro') / norm(double2q(Z), 'fro'))^2;
  if loss_U < tol && loss_V < tol && loss_UV && loss_Z< tol
     break;
  end    
end
end
