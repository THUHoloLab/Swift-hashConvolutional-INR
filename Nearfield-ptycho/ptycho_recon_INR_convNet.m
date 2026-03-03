clc
clear
reset(gpuDevice());

img_total = 75;
mag = 4;

folder = 'datas/newPath_QPT_3/';

[imgCube,imgSz]         = helpers.init_loadImgs(folder,img_total,true); pix = imgSz(1);
env                     = helpers.init_expEnvs(mag, imgSz);
[S_net,sample_cor]      = helpers.init_network(env.size, env.pixSz, 0);

D_net = [
    imageInputLayer([env.size(1),env.size(2),32],"Normalization","none");
    depthToSpace2dLayer([2,2],"Mode","CRD");
    convolution2dLayer([3,3],32,"Stride",2,"Padding","same");
    reluLayer();
    convolution2dLayer([3,3],4,"Stride",1,"Padding","same");
    ];
D_net = dlnetwork(D_net);


load([folder,'\positions.mat']);
ScanPos = [locX';locY'];


% ScanPos = ScanPos(:,1:6:end);
% imgCube = imgCube(:,:,1:6:end);

img_total = size(imgCube,3);
batchSize = 15;

epoch_max = 80;
iteration = 0;
epoch = 0;
learning_rate = 0.005;



%% recon 
global cuPtycho
cuPtycho = adCodedPtycho(mag);

optimizer_W = optimizers.AdaBelief(0.9,0.99,1e-15);
optimizer_D = optimizers.AdaBelief(0.9,0.999,1e-15);

while epoch < epoch_max
    epoch = epoch + 1;
    % cp_cube.shuffle();

    this_loss = 0;
    listBlock = randperm(img_total);
    remain = size(imgCube,3);
    start_tic = tic;
    while ~isempty(listBlock)
        iteration = iteration + 1;
        % disp(num2str(remain))

        remain = max(remain - batchSize,0);
        len = length(listBlock);

        batchIdx = listBlock(1:min(batchSize,len));
        listBlock(1:min(batchSize,len)) = [];

        scanDx = gpuArray(single(ScanPos(:,batchIdx) / pix));
        
        [loss,dldS,dldD,data,illu] = dlfeval(@model_loss, ...
                                                 S_net, ...
                                                 D_net, ...
                                                 sample_cor, ...
                                                 env.size, ...
                                                 imgCube(:,:,batchIdx), ...
                                                 scanDx, env.prop);

        this_loss = this_loss + loss;
        fprintf("at %d iter, remain = %d \n",iteration,remain)
        %% learning for parameters
        S_net = optimizer_W.step(S_net, dldS, iteration, learning_rate);
        D_net = optimizer_D.step(D_net, dldD, iteration, learning_rate);

        if mod(iteration,10) == 0
            figure(2025)
            ww = extractdata(data);
            ill = extractdata(illu);
            
            wave = ww(:,:,1) .* exp(3i .* pi .* ww(:,:,2));

            c1 = fftshift(fft2(wave));

            img_all = [mat2gray(ww(:,:,2)),mat2gray(log(abs(c1) + 1))];

            imshow(img_all,[])
            drawnow;
        end
    end

    clc
    fprintf("epoch %d done! takes %f \n",epoch,toc(start_tic));
end
% 
sz = env.size;
mkdir([folder,'results/'])
save([folder,'results/Ultnep.mat'],'S_net','D_net','sample_cor','sz');
imwrite(mat2gray(ww(:,:,2)),[folder,'results/intensity.png'])
imwrite(mat2gray(imgCube(:,:,1)),[folder,'results/raw.png'])

%% local helpers 
function [loss,...
               dldw1,dldw2,...
                           data,illu] = model_loss(S_net, ...
                                                   D_net, ...
                                                   sample_cor, ...
                                                   img_sz, ...
                                                   y_obs, ...
                                                   shifts, ...
                                                   prop)

global cuPtycho

predict = stripdims(S_net.forward(sample_cor));
predict = reshape(predict',[img_sz(1),img_sz(2),size(predict,1)]);
predict = dlarray(predict,"SSCB");
predict = real(D_net.forward(predict));

predict = sin(predict);

amp = 0.5*(predict(:,:,1) + 1);
phs =      predict(:,:,2).^4;


data = cat(3,amp,phs);
wave      = amp .* exp(6i .* pi .* phs);

illu      = predict(:,:,3) + 1i * predict(:,:,4);

dX_ds = cuPtycho(wave, illu, prop, shifts);

% loss    = l2loss(dX_ds, y_obs, "DataFormat","SSC");
loss = loss_fun.fd_loss(dX_ds,y_obs,'sum');
loss_tv = 0.1 * loss_fun.hessian_loss(wave,'anisotropic');

[dldw1, dldw2] = dlgradient(loss + loss_tv, S_net.Learnables, ...
                                            D_net.Learnables);
end