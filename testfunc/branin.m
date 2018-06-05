function y = branin(x)
%Bounds [0,1]^2
% Min = 0.1239 0.8183
% Min = 0.5428 0.1517  => 0.3979
% Min = 0.9617 0.1650
    
a = x(:,1) * 15 - 5;
b = x(:,2) * 15;

y_unscaled = (b-(5.1/(4.*pi.^2)).*a.^2+5.*a./pi-6).^2+10.*(1-1./(8.*pi)).*cos(a)+10;
y=y_unscaled/10-15;

