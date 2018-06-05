function [ g ] = gradientPosteriorMean(x, Xsamples, Ysamples, KernelMatrixInv, l, sigma)

	g = 0;
	for i = 1 :  size(KernelMatrixInv, 2)
		kstar = computeKnm(x, Xsamples, l(i,:), sigma(i));
		dkstar = -repmat(l(i,:), size(Xsamples, 1), 1) .* ...
			(repmat(x, size(Xsamples, 1), 1) - Xsamples) .* repmat(kstar', 1, size(Xsamples, 2));
		g = g + dkstar' * KernelMatrixInv{ i } * Ysamples;
	end
	g = g / size(KernelMatrixInv, 2);
