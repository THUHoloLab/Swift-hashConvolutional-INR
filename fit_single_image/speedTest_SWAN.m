% clc
clear
reset(gpuDevice());

foo = @(x) gpuArray(single(x));

img = imread("peppers.png");
% img = img(1:end,(1:size(img,1)) + 100,:);

pixels = 2048;

img = foo(img)/255;
img = imresize(img,[pixels,pixels]);

target = dlarray(reshape(img,[],3),"BC");
target_img = dlarray(img,"SSCB");

x = linspace(0,1,size(img,2));
y = linspace(0,1,size(img,1));

[x,y] = meshgrid(x,y);

pos_batch = foo([x(:),y(:)]);

pos_batch = dlarray(pos_batch,"BC");

addpath(genpath('network\'))
rootFolder = '';

high_res = pixels;

hash_level = 6;
feature_dim = 2;
E_net = network.NGP(...
    "output_dims",3,...
    "layers",0,...  % do not use MLP
    "hash_level",hash_level,...
    "bounding_box",[0,0;1,1],...
    "base_res",32,...
    "high_res",high_res,...
    "feature_dim",feature_dim,...
    "log2_hashmap_size",16,...
    "mlp_width", 32);

D_net = [
    imageInputLayer([pixels,pixels,...
                     feature_dim * hash_level],"Normalization","none");
    depthToSpace2dLayer([2,2],"Mode","CRD");
    convolution2dLayer([3,3],32,"Stride",2,"Padding","same");
    reluLayer();
    convolution2dLayer([3,3],3,"Stride",1,"Padding","same");
    ];
D_net = dlnetwork(D_net);

img_sz = size(img);
x = linspace(0,2,size(img,2));
y = linspace(0,3,size(img,1));
[x,y] = meshgrid(x,y);
pos_batch = foo([x(:),y(:)]);
pos_batch = dlarray(pos_batch,"BC");


optimizer_E = optimizers.Adam(0.9,0.99,1e-15);
optimizer_D = optimizers.Adam(0.9,0.99,1e-15);

learnRate = 0.01;

loss_data = [];
score_data = [];

%begin training
tStart = tic;
frame_counter = 0;

iter_max = 400;

t1_data = zeros(1,iter_max);
t2_data = zeros(1,iter_max);

for  iteration = 1:iter_max
    [loss,dldw1,dldw2,t1,t2] = dlfeval(@model_loss, E_net, D_net, pos_batch, [pixels,pixels],target_img);

    t1_data(1,iteration) = t1;
    t2_data(1,iteration) = t2;

    if mod(iteration,10) ==0
        [img, score] = validation(E_net,D_net,pos_batch,target_img,[pixels,pixels]);
        figure(121);
        imshow(img,[])
        score(1)
    end

    E_net = optimizer_E.step(E_net,dldw1,iteration,learnRate);
    D_net = optimizer_D.step(D_net,dldw2,iteration,learnRate);
end

mean(t1_data(1,10:end),'all')
std(t1_data(1,10:end))

mean(t2_data(1,10:end),'all')
std(t2_data(1,10:end))
imwrite(img_test,"saved_img_res = " + high_res + ".png")


function [loss,dldw1, dldw2, forward_toc, backward_toc] = model_loss(E_net, D_net, xyzs, img_sz, target)
start_timer = tic;
predict = E_net.forward(xyzs);
predict = reshape(stripdims(predict)',[img_sz(1),img_sz(2),size(predict,1)]);
predict = D_net.forward(dlarray(predict,"SSCB"));
wait(gpuDevice());
forward_toc = toc(start_timer);


loss = l2loss(predict,target);

start_timer = tic;
[dldw1,dldw2] = dlgradient(loss, E_net.Learnables, ...
                                 D_net.Learnables);
wait(gpuDevice());
backward_toc = toc(start_timer);

end



function [img, score] = validation(E_net, D_net, xyzs,target,img_sz)

predict = E_net.forward(xyzs);
predict = reshape(stripdims(predict)',[img_sz(1),img_sz(2),size(predict,1)]);
predict = D_net.forward(dlarray(predict,"SSCB"));

img = extractdata(predict);

% score = 0;
target = extractdata(target);
score = [psnr(img,target);ssim(img,target)];

% img = insertText(gather(img),[1,60],"Time:"+string(round(times)) + " s","BoxOpacity",0.5,"Font","Arial","FontSize",26);
img = insertText(gather(img),[1,10],"PSNR:"+string(round(score(1)*10)/10) + " dB","BoxOpacity",0.5,"Font","Arial","FontSize",66);
end