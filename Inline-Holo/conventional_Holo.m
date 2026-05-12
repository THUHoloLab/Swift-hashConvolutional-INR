clc
clear

addpath(genpath("optimizers"));

img = single(imread("dataset/test_out_USAF_WIN.png"))/255;


global super_reso
super_reso = 1;


img = sqrt(img);

imgSz = size(img);

get_GPU = gpuDevice();

diffractor_u = diffractor("pix_size", 2.7/2 / super_reso,...
                          "lambda",   0.532,...
                          "toz",      2520,...
                          "img_sz",   imgSz * super_reso);

global prop
prop = diffractor_u.set_propagation();

img_GT = dlarray(gpuArray(img));

canvas = single(ones(size(img_GT) * super_reso));
canvas = dlarray(gpuArray(canvas));

iter = 0;
iter_max = 500;
optimizer = optimizers.Adam(0.9,0.99,1e-15);
lr = 0.05;

%% Main training loop


while iter < iter_max
    iter = iter + 1;
    
    tic;
    [loss,grad,img_out] = dlfeval(@model_loss,...
                                   canvas,...
                                   img_GT,...
                                   prop);

    tt = toc;    
    canvas = optimizer.step(canvas,...
                                          grad,...
                                          iter,...
                                          lr);
   
    
    % canvas = min(abs(canvas),0.98) .* sign(canvas);

    fprintf("at %d-iter, takes: %4.5f, loss: %4.8f \n",iter,tt,loss);
    
    
    if mod(iter,10) == 1
        figure(121);
        imshow(imresize(abs(img_out),3),[]);
        drawnow;
    end

    if mod(iter,200) == 0
        lr = max(lr * 0.7,0.0001);
    end

end

% save('results/conven_2.mat','canvas','img_out');
% reset(gpuDevice());

function [loss,grad,img_out] = model_loss(canvas,img_GT,prop)
    global super_reso
    % f = @(x) 1./(1 + exp(-x));

    u0 = canvas;

    img_PR = abs(prop(u0)).^2;
    
    if super_reso > 1
        predict = dlresize(dlarray(img_PR,"SSCB"),"Method","nearest","Scale",1/super_reso);
    else
        predict = dlarray(img_PR,"SSCB");
    end

    loss1 = loss_fun.l2_loss(sqrt(predict), img_GT,'sum');
    tv = tv_loss(u0);

    % weight_decay =  mean(abs(canvas),'all');
    
    loss = loss1 + 0.06*tv;

    grad = dlgradient(loss, canvas);

    img_out = (extractdata(canvas));
end

function loss = tv_loss(o)
dodx = o([2:end,1],:,:) - o;
dody = o(:,[2:end,1],:) - o;

loss = abs(dodx) + abs(dody);

loss = sum(loss,'all');

end