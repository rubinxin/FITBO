function p = GMMpdf(x,Means,Variances,W)

gaussianspdf = normpdf(x,Means,sqrt(Variances));
p = sum( W.* gaussianspdf );
end