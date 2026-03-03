clc
clear

folder = 'datas/newPath_QPT_4/';

% idx = dir([folder,'imgs/']);

frame = 0;


[imgCube, imgSz] = helpers.init_loadImgs(folder,323,false);

imgCube = gpuArray(imgCube);
imgCube = imgCube ./ (mean(imgCube,3) + 1e-5);
imgCube = imgCube - min(imgCube(:));
imgCube = imgCube / max(imgCube(:));

[locX,locY,regi] = misc.track_position(imgCube);


figure(121);
plot(locX,locY,'-x','Color',[1,0,0]);

[locX,locY] = misc.refine_position(imgCube,locX,locY);
hold on;
plot(locX,locY,'-o','Color',[0,0,1]);

[locX,locY] = misc.refine_position(imgCube,locX,locY);
hold on;
plot(locX,locY,'-*','Color',[0,1,0]);


save([folder,'/positions.mat'],"locX","locY");

