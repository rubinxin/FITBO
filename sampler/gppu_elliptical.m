function [xx, cur_log_like] = gppu_elliptical(xx, chol_Sigma, log_like_fn, cur_log_like, angle_range)
%GPPU_ELLIPTICAL Gaussian prior posterior update - slice sample on random ellipses
%
%     [xx, cur_log_like] = gppu_elliptical(xx, chol_Sigma, log_like_fn[, cur_log_like])
%
% A Dx1 vector xx with prior N(0,Sigma) is updated leaving the posterior
% distribution invariant.
%
% Inputs:
%              xx Dx1 initial vector (can be any array with D elements)
%      chol_Sigma DxD chol(Sigma). Sigma is the prior covariance of xx
%     log_like_fn @fn log_like_fn(xx) returns 1x1 log likelihood
%    cur_log_like 1x1 Optional: log_like_fn(xx) of initial vector.
%                     You can omit this argument or pass [].
%     angle_range 1x1 Default 0: explore whole ellipse with break point at
%                     first rejection. Set in (0,2*pi] to explore a bracket of
%                     the specified width centred uniformly at randomly.
%
% Outputs:
%              xx Dx1 (size matches input) perturbed vector
%    cur_log_like 1x1 log_like_fn(xx) of final vector
%
% See also: GPPU_UNDERRELAX, GPPU_LINESLICE, GPPU_SPLITSLICE

% Iain Murray, September 2009

D = numel(xx);
assert(isequal(size(chol_Sigma), [D D]));
if ~exist('angle_range', 'var')
    angle_range = 0;
end
if ~exist('cur_log_like', 'var') || isempty(cur_log_like)
    cur_log_like = log_like_fn(xx);
end

% Set up the ellipse and the slice threshold
nu = reshape(chol_Sigma'*randn(D, 1), size(xx));
hh = log(rand) + cur_log_like;

% Set up a bracket of angles and pick a first proposal.
% "phi = (theta'-theta)" is a change in angle.
if angle_range <= 0
    % Bracket whole ellipse with both edges at first proposed point
    phi = rand*2*pi;
    phi_min = phi - 2*pi;
    phi_max = phi;
else
    % Randomly center bracket on current point
    phi_min = -angle_range*rand;
    phi_max = phi_min + angle_range;
    phi = rand*(phi_max - phi_min) + phi_min;
end

% Slice sampling loop
while true
    % Compute xx for proposed angle difference and check if it's on the slice
    xx_prop = xx*cos(phi) + nu*sin(phi);
    cur_log_like = log_like_fn(xx_prop);
    if cur_log_like > hh
        % New point is on slice, ** EXIT LOOP **
        break;
    end
    % Shrink slice to rejected point
    if phi > 0
        phi_max = phi;
    elseif phi < 0
        phi_min = phi;
    else
        error('BUG DETECTED: Shrunk to current position and still not acceptable.');
    end
    % Propose new angle difference
    phi = rand*(phi_max - phi_min) + phi_min;
end
xx = xx_prop;
