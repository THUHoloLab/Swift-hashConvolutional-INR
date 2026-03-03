classdef Muon < handle
    properties
        mom1
        beta1
        epsilon
        iteration
    end
    
    methods
        function self = Muon(beta1,epsilon)

            arguments
                beta1(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta1, 0), mustBeLessThan(beta1, 1)} = 0.9;
                epsilon(1,1) {mustBeNumeric, mustBeFinite, mustBePositive} = 1e-8;
            end

            self.mom1       = [];
            self.beta1      = beta1;
            self.epsilon    = epsilon;
        end

        function para = step(self,para,grad,iteration,lr)
            [para, self.mom1] = update_core(para, grad, ...
                                                  self.mom1, ...
                                                  iteration, ...
                                                  lr, ...
                                                  self.beta1, ...
                                                  self.epsilon);

        end
    end
end




function [p, avg_g] = update_core(p, g, avg_g, t, lr, beta1, epsilon)
%update_core Update parameters via adaptive moment estimation


arguments
    p
    g
    avg_g
    t(1,1) {mustBeNumeric, mustBePositive, mustBeInteger}
    
    lr(1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 0.001;
    beta1(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta1, 0), mustBeLessThan(beta1, 1)} = 0.9;
    epsilon(1,1) {mustBeNumeric, mustBeFinite, mustBePositive} = 1e-8;
end

persistent func
if isempty(func)
    func = deep.internal.LearnableUpdateFunction( ...
        @iSingleStepValue, ...
        @iSingleStepParameter);
end

if isempty(avg_g)
    % Execute a first-step update with g_av and g_sq_av set to 0.  The step
    % will create arrays for these that are the correct size
    paramArgs = {g};
    fixedArgs = {0, t, lr, beta1, epsilon};
else
    % Execute the normal update
    paramArgs = {g, matlab.lang.internal.move(avg_g)};
    fixedArgs = {t, lr, beta1, epsilon};
end

[p, avg_g] = deep.internal.networkContainerFixedArgsFun(func, ...
    p, matlab.lang.internal.move(paramArgs), fixedArgs);
end


function [p, avg_g] = iSingleStepParameter(p, g, avg_g, t, lr, beta1, epsilon)
% Apply per-parameter learn-rate factor
lr = lr .* p.LearnRateFactor;

% Short-circuit for 0 learn rate.
if lr == 0
    return;
end

[step, avg_g] = muon_step(...
    g, avg_g, lr, beta1, epsilon);

v = p.Value;
p.Value = [];
v = v + step;
p.Value = v;
end


function [p, avg_g] = iSingleStepValue(p, g, avg_g, t, lr, beta1, epsilon)
if lr == 0
    return;
end

[step, avg_g] = muon_step(...
    g, avg_g, lr, beta1, epsilon);

p = p + step;
end

function [step, avg_g] = muon_step(g, avg_g, learnrate, beta1, epsilon)
avg_g = beta1 .* avg_g + g;
avg_g_orth = NewtonSchulz(avg_g, 5, epsilon);
step = -learnrate .* avg_g_orth;
end

%% newtonschulz
function X = NewtonSchulz(G,iter,epsilon)
a =  3.4445;
b = -4.7750;
c =  2.0315;
X = G / (sqrt(sum(G.^2,'all')) + epsilon);
% X = transpose(X);
for con = 1:iter
    A = X * (X');
    B = b .* A + c .* (A * A);
    X = a .* X + B * X;
end
end