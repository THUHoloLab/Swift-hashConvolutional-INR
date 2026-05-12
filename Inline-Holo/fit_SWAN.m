% clc
clear
reset(gpuDevice());

foo = @(x) gpuArray(single(x));

img = single(imread("dataset/test_out_USAF_WIN.png"))/255;
imgSz = size(img);

img = foo(img);
img = dlarray(img,"SSCB");



global super_reso
super_reso = 1;

%% setting networks
[net_SWAN,pos_batch] = helpers.init_network(imgSz * super_reso, ...
                                            2.7/2 / super_reso, 0,"SWAN");


diffractor_u = diffractor("pix_size", 2.7/2 / super_reso,...
                          "lambda",   0.532,...
                          "toz",      2520,...
                          "img_sz",   imgSz * super_reso);

prop = diffractor_u.set_propagation();

optimizer = optimizers.Adam(0.9,0.99,1e-15);

learnRate = 0.01;

loss_data = [];
score_data = [];

%begin training
for iteration = 1:500
    
    start_timer = tic;

    tic;
    [loss,dldwE,img_out] = dlfeval(@model_loss, net_SWAN, pos_batch, img, prop);

    tt = toc;    
    this_loss = extractdata(loss);
    fprintf("at %d-iter, takes: %4.5f, loss: %4.8f \n",iteration,tt,this_loss);
    net_SWAN = optimizer.step(net_SWAN,dldwE,iteration,learnRate);

    if mod(iteration,10) == 1
        figure(123);
        imshow(abs(img_out),[]);
        drawnow;
    end

    if mod(iteration,200) == 0
        learnRate = max(learnRate * 0.7,0.0001);
    end
    loss_data = [loss_data,this_loss];
end

save('results/convNGP.mat','net_E','net_D','pos_batch','img_out');



function [loss,dldwE,img_out] = model_loss(net_E, xyzs, target, prop)
global super_reso

cav = real(net_E.forward(xyzs));

cav = (sin(cav(:,:,1)) + 1)/2 .* exp(1i * 2 * pi * sin(cav(:,:,2)).^2);

predict = abs(prop(stripdims(cav))).^2;
if super_reso > 1
    predict = dlresize(dlarray(predict,"SSCB"),"Method","nearest","Scale",1/super_reso);
else
    predict = dlarray(predict,"SSCB");
end

loss = loss_fun.l2_loss(sqrt(predict), target,'sum') + 0.06*tv_loss(cav);

dldwE = dlgradient(loss, net_E.Learnables);

img_out = extractdata(cav);
end

function loss = tv_loss(o)
dodx = o([2:end,1],:,:) - o;
dody = o(:,[2:end,1],:) - o;

loss = abs(dodx) + abs(dody);

loss = sum(loss,'all');

end