clc
clear
%% load image
foo = @(x) gpuArray(single(x));

pixels = 256;

img = imread("peppers.png");
img = foo(img)/255;
img = imresize(img,[pixels,pixels]);

target = dlarray(reshape(img,[],3),"BC");

%% networks
width = 256;
posEncoding_L = 16;
layer = [
    featureInputLayer(2 + posEncoding_L * 4,"Name","x-input")
    fullyConnectedLayer(width)
    reluLayer()
    fullyConnectedLayer(width)
    reluLayer()
    fullyConnectedLayer(width)
    reluLayer()
    fullyConnectedLayer(width)
    reluLayer()
    fullyConnectedLayer(3)
];
net = dlnetwork(layer);

%% position inputs
x = linspace(0,1,size(img,2));
y = linspace(0,1,size(img,1));
[x,y] = meshgrid(x,y);

pos_batch = foo([x(:),y(:)]);
pos_batch = dlarray(pos_batch,"BC");
optimizer_E = optimizers.Adam(0.9,0.99,1e-14);
learnRate = 0.001;

%begin training
iter_max = 500;
t1_data = zeros(1,iter_max);
t2_data = zeros(1,iter_max);


for iteration = 1:iter_max
   
    
    [loss,dldw,t1,t2] = dlfeval(@model_loss, net, pos_batch, posEncoding_L, target);
    
    if mod(iteration,10) == 0
        iteration
    end

    net = optimizer_E.step(net,dldw,iteration,learnRate);

    t1_data(1,iteration) = t1;
    t2_data(1,iteration) = t2;

end

mean(t1,'all')
std(t1)

mean(t2,'all')
std(t2')


function [loss,dldw,forward_toc,backward_toc] = model_loss(net, ...
                                                           xyzs, ...
                                                           posEncoding_L, ...
                                                           target)
% tic
start_timer = tic;
embeded = embedPositionEncoding(xyzs,posEncoding_L);
predict = net.predict(embeded);
loss = l2loss(predict,target);
wait(gpuDevice());
forward_toc = toc(start_timer);

start_timer = tic;
dldw = dlgradient(loss, net.Learnables);
wait(gpuDevice());
backward_toc = toc(start_timer);

end

% positional encoding using sin/cos functions
function embed = embedPositionEncoding(p, L)
freq_bands = reshape(repelem(2.^(0:(L-1)),1,size(p,1)),size(p,1),1,[]);

freq_bands = gpuArray(freq_bands);

emb_period = [sin(freq_bands.*repmat(p,[1,1,L]));cos(freq_bands.*repmat(p,[1,1,L]))];
emb_period = reshape(permute(stripdims(emb_period),[1,3,2]),[],size(p,2))';
emb_period = dlarray(emb_period,"BC");
embed = [p; emb_period];
end