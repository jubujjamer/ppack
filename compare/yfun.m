% A = ones(10, 5);
A= [[0.32968882, 0.45367068, 0.11706004, 0.13466604, 0.11691722];
       [0.19672058, 0.74084807, 0.14327815, 0.37836463, 0.04555327];
       [0.89998406, 0.45292057, 0.96640223, 0.72827685, 0.78236553];
       [0.67969806, 0.12748807, 0.6379347 , 0.15481726, 0.51006976];
       [0.35580545, 0.44469988, 0.98961848, 0.0373949 , 0.48300945];
       [0.20736749, 0.43999312, 0.80226303, 0.48140398, 0.1231394 ];
       [0.722152  , 0.55002954, 0.79103552, 0.99989048, 0.35846322];
       [0.09496012, 0.90680488, 0.48219842, 0.01208584, 0.02508088];
       [0.2334139 , 0.22704725, 0.86977173, 0.22217862, 0.18640152];
       [0.29088114, 0.09694667, 0.12232646, 0.12467393, 0.44656773]];


n = size(A, 2);
At = @(x) A' * x;
A = @(x) A * x;
isTruncated = false
b0 = [ones(5,1)*0.3; ones(5,1)*0.1];
m = numel(b0);                % number of measurements
% Truncated Wirtinger flow initialization
alphay = 3;                   % (4 also works fine)
y = b0.^2;                    % To be consistent with the notation in the TWF paper Algorithm 1.
lambda0 = sqrt(1/m * sum(y)); % Defined in the TWF paper Algorithm 1
idx = ones(size(b0));         % Indices of observations yi
% Truncate indices if isTruncated is true
% It discards those observations yi that are several times greater than
% the mean during spectral initialization.
if isTruncated
    idx = abs(y) <= alphay^2 * lambda0^2;
end

% Build the function handle associated to the matrix Y
% in the TWF paper Algorithm 1
% Yfunc = @(x) 1/m*At((idx.*b0.^2).*A(x));
Yfunc = @(x) 1/m*At(A(x))

% Our implemention uses Matlab's built-in function eigs() to get the leading
% eigenvector because of greater efficiency.
% Create opts struct for eigs
opts = struct;
opts.isreal = false;

% Get the eigenvector that corresponds to the largest eigenvalue of the
% associated matrix of Yfunc.
[x0, D] = eigs(Yfunc, n, k=1, 'lr', opts)
% x = ones(5, 1);
