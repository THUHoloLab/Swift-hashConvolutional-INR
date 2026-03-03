clc
clear
reset(gpuDevice());

img_total = 75;
mag = 4;

folder = 'datas/newPath_QPT_3/';

[imgCube,imgSz]         = helpers.init_loadImgs(folder,img_total,true); pix = imgSz(1);
env                     = helpers.init_expEnvs(mag, imgSz);

load([folder,'\positions.mat']);
% ScanPos = [locX';locY'];


epoch_max = 10;

wave1 = gpuArray.ones(imgSz*mag,'single');
wave2 = gpuArray.ones(imgSz*mag,'single');

iteration = 0;

for epoch = 1:epoch_max
    for pos = 1:size(imgCube,3)

        iteration = iteration + 1;
        fprintf("iter = %d \n",iteration);

        Hs = fftshift(exp(1j*2*pi.*(env.Fx .* locX(pos,1) * mag + ...
                                    env.Fy .* locY(pos,1) * mag)));
        x        = fft2(wave1);
        x_fwd    = ifft2(x .* Hs);
        x        = x_fwd .* wave2;
        x        = ifft2(fft2(x) .* env.prop);
        y_pre    = abs(x).^2;
       
        y_ds     = sqrt(imresize(y_pre,1/mag,'box'));
       
        dm = y_ds - imgCube(:,:,pos);

        x = imresize(dm ./ (y_ds + 1e-5), mag,'bicubic') .* x;

        x_bwd = ifft2(fft2(x) .* conj(env.prop));

        x = misc.deconv_pie(x_bwd, wave2, 'tPIE');

        dldw1 = ifft2(fft2(x) .* conj(Hs));
        dldw2 = misc.deconv_pie(x_bwd, x_fwd, 'tPIE');
        
        grad = tv_grad(wave1);

        wave1 = wave1 - (dldw1 + 0.0001 * grad);
        wave2 = wave2 - dldw2;

        if mod(iteration,20) == 0
            c1 = fftshift(fft2(wave1));
            img_all = [mat2gray(angle(wave1)), mat2gray(log(abs(c1) + 1))];
            figure(2023);
            imshow(img_all,[])
            drawnow;
        end
    end
end

mkdir([folder,'results/'])
save([folder,'results/rPIE.mat'],'wave1','wave2');


function grad = tv_grad(w)
dwdx = w(:,[2:end,1],:) - w;
dwdy = w([2:end,1],:,:) - w;

sss = sqrt(abs(dwdx).^2 + abs(dwdy).^2) + 1e-5;

dwdx = dwdx ./ sss;
dwdy = dwdy ./ sss;

dwdx = dwdx(:,[end,1:end-1],:) - dwdx;
dwdy = dwdy([end,1:end-1],:,:) - dwdy;

grad = dwdx + dwdy;
end

