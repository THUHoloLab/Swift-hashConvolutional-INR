%% Compare traditional blind coded ptychography and shared-INR reconstruction
% The two algorithms are shown together during iteration in the same layout
% as the final comparison figure.

clear; close all; clc;
rng(9);

%% Parameters
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

numIterTraditional = 900;
numIterInr = 900;
numIterTotal = max(numIterTraditional, numIterInr);
showEvery = 7;
noiseLevel = 0.00;

betaObj = 0.65;
betaWave = 0.20;
betaCode = 0.20;

numFreq = 6;
hiddenWidth = 256;
hiddenDepth = 3;
learningRate = 2e-3;
miniBatchScans = min(4, numScan);
phaseScaleO = 1.5 * pi;
phaseScaleC = pi;

x = ((0:N-1) - floor(N/2)) * dx;
xNorm = x / max(abs(x));
xPlot = x * 1e3;

%% Truth
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

%% Measurements
detIntensity = zeros(numScan, numCameraPixels);
for p = 1:numScan
    objShift = circshift(objTrue, [0, scanRange(p)]);
    waveAtCode = propagate1d(objShift, dx, lambda, z1);
    waveAtDet = propagate1d(codeTrue .* waveAtCode, dx, lambda, z2);
    intensity = cameraDownsample1d(abs(waveAtDet).^2, detectorBin);
    if noiseLevel > 0
        intensity = max(intensity + noiseLevel * max(intensity) * randn(size(intensity)), 0);
    end
    detIntensity(p, :) = intensity;
end

%% Traditional initialization
objTrad = ones(1, N) .* exp(1i * 0.05 * randn(1, N));
codeTrad = 0.85 * ones(1, N);
lossTrad = nan(1, numIterTraditional);

%% INR initialization on GPU
gpuDevice;
net = createSharedInrNetwork(1 + 2 * numFreq, hiddenWidth, hiddenDepth);
net = dlupdate(@gpuArray, net);

xNormTrain = gpuArray(xNorm);
detIntensityTrain = gpuArray(detIntensity);
H1 = dlarray(gpuArray(makeTransferFunction(N, dx, lambda, z1)));
H2 = dlarray(gpuArray(makeTransferFunction(N, dx, lambda, z2)));

trailingAvg = [];
trailingAvgSq = [];
lossInr = nan(1, numIterInr);

%% Joint progress window
figCompare = figure("Position",[1500,1,16*100,9*100]);
clf(figCompare);
set(figCompare, 'Color', 'w', 'Name', 'ePIE vs INR coded ptychography');

%% Synchronized iterative comparison
gifFile = 'vs_fit.gif';
gifDelayTime = 0.04;

frame = 0;
for it = 1:numIterTotal
    if it <= numIterTraditional
        [objTrad, codeTrad, lossTrad(it)] = traditionalOneIteration(objTrad, codeTrad, ...
            detIntensity, scanRange, detectorBin, dx, lambda, z1, z2, betaObj, betaWave, betaCode);
    end

    if it <= numIterInr
        batchIds = randperm(numScan, miniBatchScans);
        [loss, gradients] = dlfeval(@modelGradients, net, xNormTrain, scanRange(batchIds), ...
            detIntensityTrain(batchIds, :), detectorBin, H1, H2, numFreq, phaseScaleO, phaseScaleC);
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
            trailingAvg, trailingAvgSq, it, learningRate);
        lossInr(it) = double(gather(extractdata(loss)));
    end

    if mod(it, showEvery) == 0 || it == 1 || it == numIterTotal
        frame = frame + 1;
        objTradShow = alignScalePhase(objTrad, objTrue);
        codeTradShow = alignScalePhase(codeTrad, codeTrue);
        [objInrShow, codeInrShow] = evaluateInr(net, xNorm, numFreq, phaseScaleO, phaseScaleC);
        objInrShow = alignScalePhase(objInrShow, objTrue);
        codeInrShow = alignScalePhase(codeInrShow, codeTrue);

        plotComparisonWindow(figCompare, xPlot, objTrue, codeTrue, ...
            objTradShow, codeTradShow, objInrShow, codeInrShow, ...
            lossTrad, lossInr, it);
        pause(0.01);
        append_gif_frame(figCompare, gifFile, frame == 1, gifDelayTime);
    end
end

fprintf('ePIE final loss: %.4e\n', lossTrad(find(~isnan(lossTrad), 1, 'last')));
fprintf('INR final loss: %.4e\n', lossInr(find(~isnan(lossInr), 1, 'last')));

%% Local functions
function [objRec, codeRec, iterLoss] = traditionalOneIteration(objRec, codeRec, ...
    detIntensity, scanRange, detectorBin, dx, lambda, z1, z2, betaObj, betaWave, betaCode)
    numScan = numel(scanRange);
    order = randperm(numScan);
    iterLoss = 0;

    for kk = 1:numScan
        p = order(kk);
        s = scanRange(p);

        objShift = circshift(objRec, [0, s]);
        waveAtCode = propagate1d(objShift, dx, lambda, z1);
        exitWave = codeRec .* waveAtCode;
        waveAtDet = propagate1d(exitWave, dx, lambda, z2);

        waveAtDetNew = binnedIntensityProjection(waveAtDet, detIntensity(p, :), detectorBin);
        exitWaveNew = propagate1d(waveAtDetNew, dx, lambda, -z2);
        exitDelta = exitWaveNew - exitWave;

        waveAtCodeNew = waveAtCode ...
            + betaWave * conj(codeRec) ./ (max(abs(codeRec).^2) + eps) .* exitDelta;
        codeRec = codeRec ...
            + betaCode * conj(waveAtCode) ./ (max(abs(waveAtCode).^2) + eps) .* exitDelta;

        objShiftNew = propagate1d(waveAtCodeNew, dx, lambda, -z1);
        objCandidate = circshift(objShiftNew, [0, -s]);
        objRec = (1 - betaObj) * objRec + betaObj * objCandidate;

        predIntensity = cameraDownsample1d(abs(waveAtDet).^2, detectorBin);
        iterLoss = iterLoss + normalizedIntensityMse(predIntensity, detIntensity(p, :));
    end

    iterLoss = iterLoss / numScan;
end

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
    layers = [layers; fullyConnectedLayer(4, 'Name', 'out')];
    net = dlnetwork(layers);
end

function [loss, gradients] = modelGradients(net, xNorm, scanRangeBatch, targetBatch, ...
    detectorBin, H1, H2, numFreq, phaseScaleO, phaseScaleC)
    [obj, code] = inrForward(net, xNorm, numFreq, phaseScaleO, phaseScaleC);
    obj = stripdims(obj);
    code = stripdims(code);
    totalLoss = dlarray(gpuArray(0));

    for b = 1:numel(scanRangeBatch)
        objShift = circshift(obj, [0, scanRangeBatch(b)]);
        waveAtCode = propagate1dDl(objShift, H1);
        waveAtDet = propagate1dDl(code .* waveAtCode, H2);
        predIntensity = cameraDownsample1dDl(abs(waveAtDet).^2, detectorBin);
        target = dlarray(targetBatch(b, :));

        predIntensity = predIntensity / (mean(predIntensity, 'all') + 1e-8);
        target = target / (mean(target, 'all') + 1e-8);
        totalLoss = totalLoss + mean((predIntensity - target).^2, 'all');
    end

    loss = totalLoss / numel(scanRangeBatch);
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
    H = makeTransferFunction(numel(u1), dx, lambda, z);
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

function waveProjected = binnedIntensityProjection(waveHighRes, measuredIntensity, detectorBin)
    predBinned = cameraDownsample1d(abs(waveHighRes).^2, detectorBin);
    scaleBinned = sqrt(measuredIntensity ./ (predBinned + eps));
    waveProjected = waveHighRes .* repelem(scaleBinned, detectorBin);
end

function err = normalizedIntensityMse(predIntensity, measuredIntensity)
    predIntensity = predIntensity / (mean(predIntensity, 'all') + eps);
    measuredIntensity = measuredIntensity / (mean(measuredIntensity, 'all') + eps);
    err = mean((predIntensity - measuredIntensity).^2, 'all');
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

function plotComparisonWindow(figHandle, xPlot, objTrue, codeTrue, ...
    objTrad, codeTrad, objInr, codeInr, lossTrad, lossInr, iterNow)
    figure(figHandle);
    clf;
    tiledlayout(5, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    plot(xPlot, abs(objTrue), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, abs(objTrad), 'r--', 'LineWidth', 1.3);
    grid on; ylabel('|O|'); title(sprintf('ePIE: object amplitude, iter %d', iterNow));
    legend('Truth', 'ePIE', 'Location', 'best');
    ylim([0,1]);

    nexttile;
    plot(xPlot, abs(objTrue), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, abs(objInr), 'b--', 'LineWidth', 1.3);
    grid on; ylabel('|O|'); title(sprintf('INR: object amplitude, iter %d', iterNow));
    legend('Truth', 'INR', 'Location', 'best');
    ylim([0,1]);

    nexttile;
    plot(xPlot, unwrap(angle(objTrue)), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, unwrap(angle(objTrad)), 'r--', 'LineWidth', 1.3);
    grid on; ylabel('arg(O) / rad'); title('ePIE: object phase');
    % legend('Truth', 'ePIE', 'Location', 'best');
    ylim([-1,1]);

    nexttile;
    plot(xPlot, unwrap(angle(objTrue)), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, unwrap(angle(objInr)), 'b--', 'LineWidth', 1.3);
    grid on; ylabel('arg(O) / rad'); title('INR: object phase');
    % legend('Truth', 'INR', 'Location', 'best');
    ylim([-1,1]);

    nexttile;
    plot(xPlot, abs(codeTrue), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, abs(codeTrad), 'r--', 'LineWidth', 1.3);
    grid on; ylabel('|C|'); title('ePIE: coded surface amplitude');
    % legend('Truth', 'ePIE', 'Location', 'best');
    ylim([0.5,1]);

    nexttile;
    plot(xPlot, abs(codeTrue), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, abs(codeInr), 'b--', 'LineWidth', 1.3);
    grid on; ylabel('|C|'); title('INR: coded surface amplitude');
    % legend('Truth', 'INR', 'Location', 'best');
    ylim([0.5,1]);

    nexttile;
    plot(xPlot, unwrap(angle(codeTrue)), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, unwrap(angle(codeTrad)), 'r--', 'LineWidth', 1.3);
    grid on; ylabel('arg(C) / rad'); xlabel('x / mm'); title('ePIE: coded surface phase');
    % legend('Truth', 'ePIE', 'Location', 'best');
    ylim([0, 1]);    
    nexttile;
    plot(xPlot, unwrap(angle(codeTrue)), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, unwrap(angle(codeInr)), 'b--', 'LineWidth', 1.3);
    grid on; ylabel('arg(C) / rad'); xlabel('x / mm'); title('INR: coded surface phase');
    % legend('Truth', 'INR', 'Location', 'best');
    ylim([0, 1]);

    nexttile([1 2]);
    validTrad = find(~isnan(lossTrad));
    validInr = find(~isnan(lossInr));
    if ~isempty(validTrad)
        semilogy(validTrad, lossTrad(validTrad), 'r-', 'LineWidth', 1.5); hold on;
    end
    if ~isempty(validInr)
        semilogy(validInr, lossInr(validInr), 'b-', 'LineWidth', 1.5);
    end
    grid on; xlabel('Iteration'); ylabel('Normalized binned intensity MSE');
    title('Loss comparison');
    legend('ePIE', 'INR', 'Location', 'best');

    drawnow;
end

function append_gif_frame(fig, gifFile, isFirstFrame, delayTime)
    frame = getframe(fig);
    [img, map] = rgb2ind(frame2im(frame), 64);

    if isFirstFrame
        imwrite(img, map, gifFile, 'gif', ...
            'LoopCount', inf, ...
            'DelayTime', delayTime);
    else
        imwrite(img, map, gifFile, 'gif', ...
            'WriteMode', 'append', ...
            'DelayTime', delayTime);
    end
end
