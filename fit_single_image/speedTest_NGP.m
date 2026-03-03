% clc
clear
reset(gpuDevice());

foo = @(x) gpuArray(single(x));

img = imread("peppers.png");
% img = img(1:end,(1:size(img,1)) + 100,:);

pixels = 2048;

img = foo(img)/255;
img = imresize(img,[pixels,pixels]);

target = img;

target_reshaped = dlarray(reshape(target,[],3),"BC");

x = linspace(0,1,size(img,2));
y = linspace(0,1,size(img,1));

[x,y] = meshgrid(x,y);

pos_batch = foo([x(:),y(:)]);

pos_batch = dlarray(pos_batch,"BC");

addpath(genpath('network\'))
rootFolder = '';

high_res = pixels;

net = network.NGP(...
    "output_dims",3,...
    "layers",2,...
    "hash_level",6,...
    "bounding_box",[0,0;1,1],...
    "base_res",32,...
    "high_res",high_res,...
    "feature_dim",2,...
    "log2_hashmap_size",16,...
    "mlp_width", 64);


optimizer_E = optimizers.Adam(0.9,0.99,1e-15);

learnRate = 0.01;

loss_data = [];
score_data = [];

%begin training
tStart = tic;
frame_counter = 0;

iter_max = 500;

t1_data = zeros(1,iter_max);
t2_data = zeros(1,iter_max);
for  iteration = 1:iter_max
    [loss,dldw,t1,t2] = dlfeval(@model_loss, net, pos_batch, target_reshaped);
    
    t1_data(1,iteration) = t1;
    t2_data(1,iteration) = t2;

    if mod(iteration,10) ==0
        [img, score] = validation(net,pos_batch,target,[pixels,pixels]);
        figure(121);
        imshow(img,[])
    end

    net = optimizer_E.step(net,dldw,iteration,learnRate);
end

mean(t1_data,'all')
std(t1_data)

mean(t2_data,'all')
std(t2_data)
% imwrite(img_test,"saved_img_res = " + high_res + ".png")


function [loss,dldw,forward_toc,backward_toc] = model_loss(net, xyzs, target)

backward_toc = 0;
start_timer = tic;
predict = net.forward(xyzs);
wait(gpuDevice());
forward_toc = toc(start_timer);

loss = l2loss(predict,target);
dldw = dlgradient(loss,net.Learnables);
end



function [img, score] = validation(net,xyzs,target,img_sz)

predict = net.forward(xyzs);
img = extractdata(predict)';
img = reshape(img,[img_sz(1),img_sz(2),3]);

score = [psnr(img,target);ssim(img,target)];

% img = insertText(gather(img),[1,60],"Time:"+string(round(times)) + " s","BoxOpacity",0.5,"Font","Arial","FontSize",26);
img = insertText(gather(img),[1,10],"PSNR:"+string(round(score(1)*10)/10) + " dB","BoxOpacity",0.5,"Font","Arial","FontSize",26);
end