% clc
clear
reset(gpuDevice());

foo = @(x) gpuArray(single(x));

img = single(imread("dataset/test_out_USAF_WIN.png"));
img = mat2gray(img);

img = foo(img);
img = dlarray(img,"SSCB");

imgSz = size(img);

global super_reso
super_reso = 1;

x = linspace(0,1,size(img,2) * super_reso);
y = linspace(0,1,size(img,1) * super_reso);

[x,y] = meshgrid(x,y);

pos_batch = foo([x(:),y(:)]);

pos_batch = dlarray(pos_batch,"BC");




%% setting networks
high_res = 2048;
levels = 12; 
feature_dim = 4;

net = network.NGP(...
    "output_dims",              2,...
    "layers",                   2,...
    "hash_level",               12,...
    "bounding_box",             [0,0;1,1],...
    "base_res",                 16,...
    "high_res",                 high_res,...
    "feature_dim",              feature_dim,...
    "log2_hashmap_size",        21,...
    "MLP_width",                64);

diffractor_u = diffractor("pix_size", 2.7/2 / super_reso,...
                          "lambda",   0.532,...
                          "toz",      2520,...
                          "img_sz",   imgSz * super_reso);

prop = diffractor_u.set_propagation();

optimizer_E = optimizers.Adam(0.9,0.99,1e-15);

learnRate = 0.01;

loss_data = [];
score_data = [];

%begin training
for  iteration = 1:366
    
    start_timer = tic;

    tic;
    [loss,dldw,img_out] = dlfeval(@model_loss, net, pos_batch, img, imgSz, prop);

    tt = toc;    
    this_loss = extractdata(loss);
    fprintf("at %d-iter, takes: %4.5f, loss: %4.8f \n",iteration,tt,this_loss);
    net = optimizer_E.step(net,dldw,iteration,learnRate);

    

    figure(122);
    imshow(abs(img_out),[]);
    drawnow;

    loss_data = [loss_data,this_loss];
end


save('results/mlpNGP.mat','net','pos_batch','img_out'); 


function [loss,dldw,img_out] = model_loss(net, xyzs, target, imgSz, prop)
global super_reso

pixels = real(net.forward(xyzs));

cav = pixels(1,:) + 1i * pixels(2,:);
cav = reshape(stripdims(cav)',imgSz(1)*super_reso,imgSz(2)*super_reso);

predict = abs(prop(cav)).^2;
if super_reso > 1
    predict = dlresize(dlarray(predict,"SSCB"),"Method","nearest","Scale",1/super_reso);
else
    predict = dlarray(predict,"SSCB");
end

loss = loss_fun.l2_loss(sqrt(predict), target,'sum') + 0.06*tv_loss(cav);

dldw = dlgradient(loss, net.Learnables);

img_out = extractdata(cav);
end


function loss = tv_loss(o)
dodx = o([2:end,1],:,:) - o;
dody = o(:,[2:end,1],:) - o;
loss = abs(dodx) + abs(dody);
loss = sum(loss,'all');
end