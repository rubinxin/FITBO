function [ f ] = posteriorMean(x, Xsamples, Ysamples, KernelMatrixInv, l, sigma)

	f = 0;
	for i = 1 : size(KernelMatrixInv, 2)
		kstar = computeKnm(x, Xsamples, l(i,:), sigma(i));
		f = f + kstar * KernelMatrixInv{ i } * Ysamples;
	end
	f = f / size(KernelMatrixInv, 2);
