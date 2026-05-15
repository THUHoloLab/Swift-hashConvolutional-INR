%% 1D blind coded ptychography with shared INR using MATLAB dlnetwork
% Forward model:
%   O_s(x)   = O(x - shift)
%   U_c(x)   = Prop_z1{O_s(x)}
%   V_c(x)   = C(x) .* U_c(x)
%   U_det(x) = Prop_z2{V_c(x)}
%   I_s[m]   = sum_{x in camera pixel m} abs(U_det(x)).^2
%
% Reconstruction:
%   x -> sinusoidal positional encoding -> shared MLP trunk -> [O head, C head]
%   O and C are inferred once per gradient step; object shifts use circshift.
% The MLP is a MATLAB dlnetwork. Gradients and Adam updates use built-in
% Deep Learning Toolbox functions. Training arrays are moved to gpuArray.

clear; close all; clc;
rng(8);

%% Physical and numerical parameters
N = 512;
dx = 6.5e-6;
lambda = 532e-9;
z1 = 835e-3;
z2 = 355e-3;

detectorBin = 4;
numCameraPixels = N / detectorBin;
if mod(N, detectorBin) ~= 0
    error('N must be divisible by detectorBin.');
end

scanStep = 6;
scanRange = -64:scanStep:64;
numScan = numel(scanRange);

numIter = 900;
learningRate = 2e-3;
miniBatchScans = min(4, numScan);
noiseLevel = 0.00;
showEvery = 20;

numFreq = 5;
hiddenWidth = 256;
hiddenDepth = 3;
phaseScaleO = 1.5 * pi;
phaseScaleC = pi;

x = ((0:N-1) - floor(N/2)) * dx;
xNorm = x / max(abs(x));
xPlot = x * 1e3;

gpuDevice;

%% Ground truth O and C, only for simulation and visual comparison
ampTrue = 0.25 ...
    + 0.48 * exp(-((x + 0.95e-3) / 0.22e-3).^2) ...
    + 0.36 * exp(-((x - 0.08e-3) / 0.38e-3).^2) ...
    + 0.30 * exp(-((x - 0.72e-3) / 0.18e-3).^2);
ampTrue = ampTrue + 0.08 * sin(6 * pi * x / 0.42e-3).^2;
ampTrue = normalizeToRange(ampTrue, 0.20, 1.00);

phaseTrue = 0.9 * sin(2 * pi * x / 1.15e-3) ...
    + 0.75 * exp(-((x + 0.35e-3) / 0.28e-3).^2) ...
    - 0.65 * exp(-((x - 0.52e-3) / 0.22e-3).^2);
objTrue = ampTrue .* exp(1i * phaseTrue);

codeAmpTrue = 0.55 + 0.45 * imresize(rand(1, round(N/10)), [1, N], 'box');
codePhaseTrue = imresize(rand(1, round(N/3)), [1, N], 'box');
codeTrue = codeAmpTrue .* exp(1i * codePhaseTrue);

%% Simulate binned camera measurements
detIntensity = zeros(numScan, numCameraPixels);

for p = 1:numScan
    s = scanRange(p);
    objShift = circshift(objTrue, [0, s]);
    waveAtCode = propagate1d(objShift, dx, lambda, z1);
    exitWave = codeTrue .* waveAtCode;
    waveAtDet = propagate1d(exitWave, dx, lambda, z2);

    intensity = cameraDownsample1d(abs(waveAtDet).^2, detectorBin);
    if noiseLevel > 0
        intensity = intensity + noiseLevel * max(intensity) * randn(size(intensity));
        intensity = max(intensity, 0);
    end
    detIntensity(p, :) = intensity;
end

%% Shared INR network
inputDim = 1 + 2 * numFreq;
net = createSharedInrNetwork(inputDim, hiddenWidth, hiddenDepth);
net = dlupdate(@gpuArray, net);

xNormTrain = gpuArray(xNorm);
detIntensityTrain = gpuArray(detIntensity);
H1 = dlarray(gpuArray(makeTransferFunction(N, dx, lambda, z1)));
H2 = dlarray(gpuArray(makeTransferFunction(N, dx, lambda, z2)));

lossHist = zeros(1, numIter);
ampHist = zeros(numIter, N);
phaseHist = zeros(numIter, N);
codeAmpHist = zeros(numIter, N);
codePhaseHist = zeros(numIter, N);

trailingAvg = [];
trailingAvgSq = [];

figNum = 10;
fig = figure(figNum);
clf(fig);
set(fig, 'Color', 'w', 'Name', 'Shared INR blind coded ptychography');

%% Optimization
for it = 1:numIter
    batchIds = randperm(numScan, miniBatchScans);

    [loss, gradients] = dlfeval(@modelGradients, net, xNormTrain, scanRange(batchIds), ...
        detIntensityTrain(batchIds, :), detectorBin, H1, H2, numFreq, ...
        phaseScaleO, phaseScaleC);

    [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
        trailingAvg, trailingAvgSq, it, learningRate);

    lossHist(it) = double(gather(extractdata(loss)));

    if mod(it, showEvery) == 0 || it == 1 || it == numIter
        [objRec, codeRec] = evaluateInr(net, xNorm, numFreq, phaseScaleO, phaseScaleC);
        objAligned = alignScalePhase(objRec, objTrue);
        codeAligned = alignScalePhase(codeRec, codeTrue);

        ampHist(it, :) = abs(objAligned);
        phaseHist(it, :) = angle(objAligned);
        codeAmpHist(it, :) = abs(codeAligned);
        codePhaseHist(it, :) = angle(codeAligned);

        plotInrProgress(figNum, xPlot, objTrue, objAligned, codeTrue, codeAligned, ...
            ampHist, phaseHist, codeAmpHist, codePhaseHist, lossHist, it);
        pause(0.01);
    end
end

%% Final display
[objRec, codeRec] = evaluateInr(net, xNorm, numFreq, phaseScaleO, phaseScaleC);
objAligned = alignScalePhase(objRec, objTrue);
codeAligned = alignScalePhase(codeRec, codeTrue);

figure(11);
clf;
set(gcf, 'Color', 'w', 'Name', 'Final shared INR result');
tiledlayout(5, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
plot(xPlot, abs(objTrue), 'k-', 'LineWidth', 1.6); hold on;
plot(xPlot, abs(objAligned), 'r--', 'LineWidth', 1.4);
grid on; ylabel('|O|'); legend('True O', 'INR O', 'Location', 'best');
title('Object amplitude');

nexttile;
plot(xPlot, unwrap(angle(objTrue)), 'k-', 'LineWidth', 1.6); hold on;
plot(xPlot, unwrap(angle(objAligned)), 'r--', 'LineWidth', 1.4);
grid on; ylabel('arg(O) / rad'); legend('True O', 'INR O', 'Location', 'best');
title('Object phase');

nexttile;
plot(xPlot, abs(codeTrue), 'k-', 'LineWidth', 1.6); hold on;
plot(xPlot, abs(codeAligned), 'r--', 'LineWidth', 1.4);
grid on; ylabel('|C|'); legend('True C', 'INR C', 'Location', 'best');
title('Coded surface amplitude');

nexttile;
plot(xPlot, unwrap(angle(codeTrue)), 'k-', 'LineWidth', 1.6); hold on;
plot(xPlot, unwrap(angle(codeAligned)), 'r--', 'LineWidth', 1.4);
grid on; ylabel('arg(C) / rad'); legend('True C', 'INR C', 'Location', 'best');
title('Coded surface phase');

nexttile;
semilogy(lossHist, 'b-', 'LineWidth', 1.5);
grid on; xlabel('Iteration'); ylabel('Binned intensity MSE');
title('Gradient descent loss');

fprintf('Done. Final INR loss: %.4e\n', lossHist(end));

%% Local functions
function net = createSharedInrNetwork(inputDim, hiddenWidth, hiddenDepth)
    layers = [
        featureInputLayer(inputDim, 'Normalization', 'none', 'Name', 'x_pe')
        fullyConnectedLayer(hiddenWidth, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
    ];

    for k = 2:hiddenDepth
        layers = [
            layers
            fullyConnectedLayer(hiddenWidth, 'Name', sprintf('fc%d', k))
            reluLayer('Name', sprintf('relu%d', k))
        ]; %#ok<AGROW>
    end

    layers = [
        layers
        fullyConnectedLayer(4, 'Name', 'out')
    ];

    net = dlnetwork(layers);
end

function [loss, gradients] = modelGradients(net, xNorm, scanRangeBatch, targetBatch, ...
    detectorBin, H1, H2, numFreq, phaseScaleO, phaseScaleC)
    [obj, code] = inrForward(net, xNorm, numFreq, phaseScaleO, phaseScaleC);
    obj = stripdims(obj);
    code = stripdims(code);

    numBatch = numel(scanRangeBatch);
    totalLoss = dlarray(gpuArray(0));

    for b = 1:numBatch
        objShift = circshift(obj, [0, scanRangeBatch(b)]);
        waveAtCode = propagate1dDl(objShift, H1);
        exitWave = code .* waveAtCode;
        waveAtDet = propagate1dDl(exitWave, H2);

        predIntensity = cameraDownsample1dDl(abs(waveAtDet).^2, detectorBin);
        target = dlarray(targetBatch(b, :));

        predIntensity = predIntensity / (mean(predIntensity, 'all') + 1e-8);
        target = target / (mean(target, 'all') + 1e-8);
        totalLoss = totalLoss + mean((predIntensity - target).^2, 'all');
    end

    loss = totalLoss / numBatch;
    gradients = dlgradient(loss, net.Learnables);
end

function [obj, code] = inrForward(net, xNorm, numFreq, phaseScaleO, phaseScaleC)
    pe = positionalEncoding(xNorm, numFreq);
    dlX = dlarray(pe, 'CB');
    y = real(forward(net, dlX));

    ampO = (sin(y(1, :)) + 1)/2;
    phaseO = phaseScaleO * sin(y(2, :));
    obj = ampO .* exp(1i * phaseO);

    ampC = (sin(y(3, :)) + 1)/2;
    phaseC = phaseScaleC * sin(y(4, :));
    code = ampC .* exp(1i * phaseC);
end

function pe = positionalEncoding(xNorm, numFreq)
    pe = xNorm;
    for k = 0:(numFreq - 1)
        f = 2^k * pi;
        pe = [pe; sin(f * xNorm); cos(f * xNorm)]; %#ok<AGROW>
    end
end

function u2 = propagate1d(u1, dx, lambda, z)
    n = numel(u1);
    H = makeTransferFunction(n, dx, lambda, z);
    u2 = ifft(ifftshift(fftshift(fft(u1)) .* H));
end

function u2 = propagate1dDl(u1, H)
    u2 = ifft(ifftshift(fftshift(fft(u1)) .* H));
end

function H = makeTransferFunction(n, dx, lambda, z)
    fx = ((0:n-1) - floor(n/2)) / (n * dx);
    H = exp(-1i * pi * lambda * z * fx.^2);
end

function y = cameraDownsample1d(intensityHighRes, detectorBin)
    numCameraPixels = numel(intensityHighRes) / detectorBin;
    y = sum(reshape(intensityHighRes, detectorBin, numCameraPixels), 1);
end

function y = cameraDownsample1dDl(intensityHighRes, detectorBin)
    numCameraPixels = numel(intensityHighRes) / detectorBin;
    y = sum(reshape(intensityHighRes, detectorBin, numCameraPixels), 1);
end

function [objRec, codeRec] = evaluateInr(net, xNorm, numFreq, phaseScaleO, phaseScaleC)
    [objDl, codeDl] = inrForward(net, gpuArray(xNorm), numFreq, phaseScaleO, phaseScaleC);
    objRec = gather(extractdata(objDl));
    codeRec = gather(extractdata(codeDl));
end

function y = normalizeToRange(y, ymin, ymax)
    y = y - min(y);
    y = y ./ (max(y) + eps);
    y = ymin + (ymax - ymin) * y;
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function estAligned = alignScalePhase(est, ref)
    alpha = sum(conj(est) .* ref) / (sum(abs(est).^2) + eps);
    estAligned = alpha * est;
end

function plotInrProgress(figNum, xPlot, objTrue, objRec, codeTrue, codeRec, ...
    ampHist, phaseHist, codeAmpHist, codePhaseHist, lossHist, it)
    figure(figNum);
    clf;
    tiledlayout(5, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    valid = any(ampHist ~= 0, 2);
    validIter = find(valid);

    nexttile;
    plot(xPlot, abs(objTrue), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, abs(objRec), 'r--', 'LineWidth', 1.3);
    grid on; xlim([xPlot(1), xPlot(end)]);
    ylabel('|O|'); title(sprintf('INR O amplitude, iteration %d', it));
    legend('True O', 'INR O', 'Location', 'best');

    nexttile;
    imagesc(xPlot, validIter, ampHist(valid, :));
    axis xy; colorbar; xlabel('x / mm'); ylabel('Iteration');
    title('O amplitude evolution');

    nexttile;
    plot(xPlot, unwrap(angle(objTrue)), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, unwrap(angle(objRec)), 'r--', 'LineWidth', 1.3);
    grid on; xlim([xPlot(1), xPlot(end)]);
    ylabel('arg(O) / rad'); title('INR O phase');
    legend('True O', 'INR O', 'Location', 'best');

    nexttile;
    imagesc(xPlot, validIter, unwrap(phaseHist(valid, :), [], 2));
    axis xy; colorbar; xlabel('x / mm'); ylabel('Iteration');
    title('O phase evolution');

    nexttile;
    plot(xPlot, abs(codeTrue), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, abs(codeRec), 'r--', 'LineWidth', 1.3);
    grid on; xlim([xPlot(1), xPlot(end)]);
    ylabel('|C|'); title('INR C amplitude');
    legend('True C', 'INR C', 'Location', 'best');

    nexttile;
    imagesc(xPlot, validIter, codeAmpHist(valid, :));
    axis xy; colorbar; xlabel('x / mm'); ylabel('Iteration');
    title('C amplitude evolution');

    nexttile;
    plot(xPlot, unwrap(angle(codeTrue)), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, unwrap(angle(codeRec)), 'r--', 'LineWidth', 1.3);
    grid on; xlim([xPlot(1), xPlot(end)]);
    ylabel('arg(C) / rad'); title('INR C phase');
    legend('True C', 'INR C', 'Location', 'best');

    nexttile;
    imagesc(xPlot, validIter, unwrap(codePhaseHist(valid, :), [], 2));
    axis xy; colorbar; xlabel('x / mm'); ylabel('Iteration');
    title('C phase evolution');

    nexttile([1 2]);
    semilogy(1:it, lossHist(1:it), 'b-', 'LineWidth', 1.5);
    grid on; xlabel('Iteration'); ylabel('MSE');
    title('Gradient descent loss');

    drawnow;
end
