function [ x_next ] = Optimization(target, lb, hb,guesses)

	d = size(lb, 2);

	% We evaluate the objective in a grid and pick the best location

	gridSize = 1000;

% 	Xgrid = [ Xgrid ; guesses ];
    X_test   = repmat(lb, gridSize, 1) + repmat((hb - lb), gridSize, 1) .* rand(gridSize, d);
    X_test   = [X_test; guesses];
    
	y = target(X_test);

	[ minValue, minIndex ] = min(y);

	start = X_test(minIndex,:);

	% We optimize starting at the best location
    [x_next, global_minValue] = fmincon(target, start, [], [], [], [], lb, hb,[],...
                optimset('Display', 'off'));

	

% 	hessian = zeros(d * (d - 1) / 2, 1);
% 	counter = 1;
% 	if (d > 1)
% 		j = 1;
% 		while (j <= d)
% 			h = j + 1;
% 			while (h <= d)
% 				hessian(counter) = retHessian( j, h );
% 				counter = counter + 1;
% 				h = h + 1;
% 			end
% 			j = j + 1;
% 		end
% 	end

end
