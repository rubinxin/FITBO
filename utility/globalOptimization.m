function [ optimum hessian ] = globalOptimization(target, gradient, xmin, xmax, guesses)

	d = size(xmin, 2);

	% We evaluate the objective in a grid and pick the best location

	gridSize = 1000;

	Xgrid = repmat(xmin, gridSize, 1) + repmat((xmax - xmin), gridSize, 1) .* rand(gridSize, d);
	Xgrid = [ Xgrid ; guesses ];

	y = target(Xgrid);

	[ minValue minIndex ] = min(y);

	start = Xgrid(minIndex,:);

	targetOptimization = @(x) deal(target(x), gradient(x));

	% We optimize starting at the best location

	[ optimum, fval, exitflag, output, lambda, grad, retHessian] = fmincon(targetOptimization, start, [], [], [], [], xmin, xmax, [], ...
                optimset('MaxFunEvals', 100, 'TolX', eps, 'Display', 'off', 'GradObj', 'on'));

	hessian = zeros(d * (d - 1) / 2, 1);
	counter = 1;
	if (d > 1)
		j = 1;
		while (j <= d)
			h = j + 1;
			while (h <= d)
				hessian(counter) = retHessian( j, h );
				counter = counter + 1;
				h = h + 1;
			end
			j = j + 1;
		end
	end

end
