clc
clear
reset(gpuDevice());

img_total = 75;
mag = 4;

folder = 'datas/newPath_QPT_3/';

[imgCube,imgSz]         = helpers.init_loadImgs(folder,img_total,true); pix = imgSz(1);
env                     = helpers.init_expEnvs(mag, imgSz);
[sample_net,sample_cor] = helpers.init_network(env.size, env.pixSz,2);

load([folder,'\positions.mat']);
ScanPos = [locX';locY'];
save imgCube imgCube
% cp_cube = combine(arrayDatastore(ScanPos, 'IterationDimension',1), ...
                  % arrayDatastore(imgCube, 'IterationDimension',3));

img_total = size(imgCube,3);
batchSize = 15;

epoch_max = 100;
iteration = 0;
epoch = 0;

%% recon 
lr = 0.005;

global cuPtycho
cuPtycho = adCodedPtycho(mag);

optimizer_W = optimizers.AdaBelief(0.9,0.99,1e-15);

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
        [loss,dldw1,data,illu] = dlfeval(@model_loss,...
                                                 sample_net,...
                                                 sample_cor,...
                                                 env.size,...
                                                 imgCube(:,:,batchIdx), ...
                                                 scanDx, env.prop);

        this_loss = this_loss + loss;
        fprintf("at %d iter, remain = %d \n",iteration,remain)
        %% learning for parameters
        sample_net = optimizer_W.step(sample_net, dldw1, iteration, lr);

        if mod(iteration,10) == 0
            figure(2024)
            ww = extractdata(data);
            ill = extractdata(illu);

            img_all = [mat2gray(ww(:,:,1)),mat2gray(ww(:,:,2));
                       mat2gray(abs(ill).^2), mat2gray(angle(ill))];

            imshow(img_all,[])
            drawnow;
        end
    end

    clc
    fprintf("epoch %d done! takes %f \n",epoch,toc(start_tic));
end


function [loss,dldw1,data,illu] = model_loss(sample_net,...
                                             sample_cor,...
                                             img_sz,...
                                             y_obs, ...
                                             shifts,prop)


global cuPtycho

predict = real(stripdims(sample_net.forward(sample_cor)));
predict = reshape(predict',[img_sz(1),img_sz(2),size(predict,1)]);
predict = sin(predict);

amp = 0.5*(predict(:,:,1) + 1);
phs =      predict(:,:,2).^2;

% amp = predict(:,:,1);
% phs = predict(:,:,2);

data = cat(3,amp,phs);

wave      = amp .* exp(6i .* pi .* phs);
illu      = predict(:,:,3) + 1i * predict(:,:,4);
% illu0 = complex(gpuArray.ones(size(illu),'single'));

dX_ds = cuPtycho(wave, illu, prop, shifts);

% loss    = huber(dX_ds, y_obs, "DataFormat","SSB");
loss = loss_fun.fd_loss(dX_ds,y_obs,'sum');
loss_tv = 0.1 * loss_fun.hessian_loss(wave,'anisotropic');

dldw1 = dlgradient(loss + loss_tv, sample_net.Learnables);
end