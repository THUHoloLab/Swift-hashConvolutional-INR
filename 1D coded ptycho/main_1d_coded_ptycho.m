%% 1D blind coded ptychography with two propagation distances
% Forward model for scan position s:
%   O_s(x)   = shifted object
%   U_c(x)   = Prop_z1{O_s(x)}
%   V_c(x)   = C(x) .* U_c(x)
%   U_det(x) = Prop_z2{V_c(x)}
%   I_s[m]   = sum_{x in camera pixel m} abs(U_det(x)).^2
%
% In reconstruction, both O and the coded surface C are unknown.

clear; close all; clc;
rng(7);

%% Physical and numerical parameters
N = 512;                 % 1D sampling number
dx = 6.5e-6;             % sample pitch / m
lambda = 532e-9;         % wavelength / m
z1 = 835e-3;               % object to coded surface distance / m
z2 = 355e-3;              % coded surface to detector distance / m
detectorBin = 4;          % camera pixel size in simulation-grid samples
numCameraPixels = N / detectorBin;
if mod(N, detectorBin) ~= 0
    error('N must be divisible by detectorBin.');
end

scanStep = 6;           % integer object translation step / pixels
scanRange = -64:scanStep:64;
numScan = numel(scanRange);

numIter = 120;
betaObj = 0.65;          % object update relaxation
betaWave = 0.70;         % incident wave update at C
betaCode = 0.30;         % coded surface update relaxation
noiseLevel = 0.00;       % try 0.005 or 0.01 for noisy data
showEvery = 1;
pauseTime = 0.01;

x = ((0:N-1) - floor(N/2)) * dx;
xPlot = x * 1e3;

%% Ground truth object O and coded surface C, used only to simulate data
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

codeAmpTrue = 0.55 + 0.45 * imresize(rand(1, round(N/10)),[1,N],'box');
codePhaseTrue = imresize(rand(1, round(N/3)),[1,N],'box');
codeTrue = codeAmpTrue .* exp(1i * codePhaseTrue);

%% Simulate coded ptychography measurements with camera downsampling
detIntensity = zeros(numScan, numCameraPixels);

for p = 1:numScan
    s = scanRange(p);
    objShift = shift1d(objTrue, s);

    waveAtCode = propagate1d(objShift, dx, lambda, z1);
    exitWave = codeTrue .* waveAtCode;
    waveAtDet = propagate1d(exitWave, dx, lambda, z2);

    intensityHighRes = abs(waveAtDet).^2;
    intensity = cameraDownsample1d(intensityHighRes, detectorBin);
    if noiseLevel > 0
        intensity = intensity + noiseLevel * max(intensity) * randn(size(intensity));
        intensity = max(intensity, 0);
    end

    detIntensity(p, :) = intensity;
end

%% Blind iterative reconstruction of O and C
objRec = ones(1, N);
% objRec = objRec + 0.02 * (randn(1, N) + 1i * randn(1, N));

codeRec = 0.85 * ones(1, N);

err = zeros(1, numIter);
ampHist = zeros(numIter, N);
phaseHist = zeros(numIter, N);
codeAmpHist = zeros(numIter, N);
codePhaseHist = zeros(numIter, N);

figNum = 1;
fig = figure(figNum);
clf(fig);
set(fig, 'Color', 'w', 'Name', '1D blind coded ptychography: O shift, z1, C, z2');
% fig.Position(3:4) = [1220, 780];

for it = 1:numIter
    order = randperm(numScan);
    iterErr = 0;

    for kk = 1:numScan
        p = order(kk);
        s = scanRange(p);

        objShift = shift1d(objRec, s);

        % Forward propagation: O_s -> z1 -> unknown C -> z2 -> detector.
        waveAtCode = propagate1d(objShift, dx, lambda, z1);
        exitWave = codeRec .* waveAtCode;
        waveAtDet = propagate1d(exitWave, dx, lambda, z2);

        % Camera-plane projection with pixel binning.
        waveAtDetNew = binnedIntensityProjection(waveAtDet, detIntensity(p, :), detectorBin);

        % Back to the coded surface.
        exitWaveNew = propagate1d(waveAtDetNew, dx, lambda, -z2);
        exitDelta = exitWaveNew - exitWave;

        % Blind ePIE-style updates for incident wave and coded surface.
        waveAtCodeNew = waveAtCode ...
            + betaWave * conj(codeRec) ./ (max(abs(codeRec).^2) + eps) .* exitDelta;

        codeRec = codeRec ...
            + betaCode * conj(waveAtCode) ./ (max(abs(waveAtCode).^2) + eps) .* exitDelta;
        % codeRec = constrainCode(codeRec);

        % Back to shifted object plane, then shift update to common O coordinate.
        objShiftNew = propagate1d(waveAtCodeNew, dx, lambda, -z1);
        objCandidate = shift1d(objShiftNew, -s);
        objRec = (1 - betaObj) * objRec + betaObj * objCandidate;

        predIntensity = cameraDownsample1d(abs(waveAtDet).^2, detectorBin);
        iterErr = iterErr + norm(sqrt(predIntensity) - sqrt(detIntensity(p, :))) ...
            / (norm(sqrt(detIntensity(p, :))) + eps);
    end

    err(it) = iterErr / numScan;

    objAligned = alignScalePhase(objRec, objTrue);
    codeAligned = alignScalePhase(codeRec, codeTrue);
    ampHist(it, :) = abs(objAligned);
    phaseHist(it, :) = angle(objAligned);
    codeAmpHist(it, :) = abs(codeAligned);
    codePhaseHist(it, :) = angle(codeAligned);

    if mod(it, showEvery) == 0 || it == 1 || it == numIter
        plotReconstruction(figNum, xPlot, objTrue, objAligned, codeTrue, codeAligned, ...
            ampHist, phaseHist, codeAmpHist, codePhaseHist, err, it);
        pause(pauseTime);
    end
end

%% Final display
objAligned = alignScalePhase(objRec, objTrue);
codeAligned = alignScalePhase(codeRec, codeTrue);

figure(2);
clf;
set(gcf, 'Color', 'w', 'Name', 'Final 1D blind coded ptychography result');
tiledlayout(5, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
plot(xPlot, abs(objTrue), 'k-', 'LineWidth', 1.6); hold on;
plot(xPlot, abs(objAligned), 'r--', 'LineWidth', 1.4);
grid on; ylabel('|O|');
legend('True O', 'Recovered O', 'Location', 'best');
title('Recovered object amplitude');

nexttile;
plot(xPlot, unwrap(angle(objTrue)), 'k-', 'LineWidth', 1.6); hold on;
plot(xPlot, unwrap(angle(objAligned)), 'r--', 'LineWidth', 1.4);
grid on; ylabel('arg(O) / rad');
legend('True O', 'Recovered O', 'Location', 'best');
title('Recovered object phase');

nexttile;
plot(xPlot, abs(codeTrue), 'k-', 'LineWidth', 1.6); hold on;
plot(xPlot, abs(codeAligned), 'r--', 'LineWidth', 1.4);
grid on; ylabel('|C|');
legend('True C', 'Recovered C', 'Location', 'best');
title('Recovered coded surface amplitude');

nexttile;
plot(xPlot, unwrap(angle(codeTrue)), 'k-', 'LineWidth', 1.6); hold on;
plot(xPlot, unwrap(angle(codeAligned)), 'r--', 'LineWidth', 1.4);
grid on; ylabel('arg(C) / rad');
legend('True C', 'Recovered C', 'Location', 'best');
title('Recovered coded surface phase');

nexttile;
semilogy(err, 'b-', 'LineWidth', 1.5);
grid on; xlabel('Iteration'); ylabel('Detector amplitude error');
title('Convergence');

fprintf('Done. Final detector amplitude error: %.4e\n', err(end));

%% Local functions
function u2 = propagate1d(u1, dx, lambda, z)
    % 1D Fresnel transfer-function propagation.
    n = numel(u1);
    fx = ((0:n-1) - floor(n/2)) / (n * dx);
    H = exp(-1i * pi * lambda * z * fx.^2);
    u2 = ifft(ifftshift(fftshift(fft(u1)) .* H));
end

function y = shift1d(x, pixelShift)
    % Integer translation of the object along x.
    y = circshift(x, [0, pixelShift]);
end

function y = cameraDownsample1d(intensityHighRes, detectorBin)
    % Camera pixels integrate high-resolution detector intensity.
    numCameraPixels = numel(intensityHighRes) / detectorBin;
    y = sum(reshape(intensityHighRes, detectorBin, numCameraPixels), 1);
end

function waveProjected = binnedIntensityProjection(waveHighRes, measuredIntensity, detectorBin)
    % Project a high-resolution detector field onto binned camera intensities.
    % Each camera pixel constrains the sum of intensities in one detector bin.
    predIntensity = abs(waveHighRes).^2;
    predBinned = cameraDownsample1d(predIntensity, detectorBin);
    scaleBinned = sqrt(measuredIntensity ./ (predBinned + eps));
    scaleHighRes = repelem(scaleBinned, detectorBin);
    waveProjected = waveHighRes .* scaleHighRes;
end

function codeOut = constrainCode(codeIn)
    % Weak transmission constraint; it also reduces O-C scale ambiguity.
    amp = abs(codeIn);
    ph = angle(codeIn);
    amp = min(max(amp, 0.05), 1.20);
    codeOut = amp .* exp(1i * ph);
end

function y = normalizeToRange(y, ymin, ymax)
    y = y - min(y);
    y = y ./ (max(y) + eps);
    y = ymin + (ymax - ymin) * y;
end

function estAligned = alignScalePhase(est, ref)
    alpha = sum(conj(est) .* ref) / (sum(abs(est).^2) + eps);
    estAligned = alpha * est;
end

function plotReconstruction(figNum, xPlot, objTrue, objRec, codeTrue, codeRec, ...
    ampHist, phaseHist, codeAmpHist, codePhaseHist, err, it)
    figure(figNum);
    clf;
    tiledlayout(5, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    plot(xPlot, abs(objTrue), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, abs(objRec), 'r--', 'LineWidth', 1.3);
    grid on; xlim([xPlot(1), xPlot(end)]);
    ylabel('|O|');
    title(sprintf('O amplitude, iteration %d', it));
    legend('True O', 'Recovered O', 'Location', 'best');

    nexttile;
    imagesc(xPlot, 1:it, ampHist(1:it, :));
    axis xy; colorbar;
    xlabel('x / mm'); ylabel('Iteration');
    title('O amplitude evolution');

    nexttile;
    plot(xPlot, unwrap(angle(objTrue)), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, unwrap(angle(objRec)), 'r--', 'LineWidth', 1.3);
    grid on; xlim([xPlot(1), xPlot(end)]);
    ylabel('arg(O) / rad');
    title('O phase');
    legend('True O', 'Recovered O', 'Location', 'best');

    nexttile;
    imagesc(xPlot, 1:it, unwrap(phaseHist(1:it, :), [], 2));
    axis xy; colorbar;
    xlabel('x / mm'); ylabel('Iteration');
    title('O phase evolution');

    nexttile;
    plot(xPlot, abs(codeTrue), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, abs(codeRec), 'r--', 'LineWidth', 1.3);
    grid on; xlim([xPlot(1), xPlot(end)]);
    ylabel('|C|');
    title('C amplitude');
    legend('True C', 'Recovered C', 'Location', 'best');

    nexttile;
    imagesc(xPlot, 1:it, codeAmpHist(1:it, :));
    axis xy; colorbar;
    xlabel('x / mm'); ylabel('Iteration');
    title('C amplitude evolution');

    nexttile;
    plot(xPlot, unwrap(angle(codeTrue)), 'k-', 'LineWidth', 1.5); hold on;
    plot(xPlot, unwrap(angle(codeRec)), 'r--', 'LineWidth', 1.3);
    grid on; xlim([xPlot(1), xPlot(end)]);
    ylabel('arg(C) / rad');
    title('C phase');
    legend('True C', 'Recovered C', 'Location', 'best');

    nexttile;
    imagesc(xPlot, 1:it, unwrap(codePhaseHist(1:it, :), [], 2));
    axis xy; colorbar;
    xlabel('x / mm'); ylabel('Iteration');
    title('C phase evolution');

    nexttile([1 2]);
    semilogy(1:it, err(1:it), 'b-', 'LineWidth', 1.5);
    grid on; xlim([1, max(2, it)]);
    xlabel('Iteration');
    ylabel('Detector amplitude error');
    title('Convergence curve');

    drawnow;
end
