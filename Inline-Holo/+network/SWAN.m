function net = SWAN(args)
    arguments
        args.output_dims = 2;
        args.layers = 2;
        args.hash_level = 8;
        args.base_res = 16;
        args.high_res = 4096;
        args.bounding_box = [0,0;1,1];
        args.feature_dim = 4;
        args.log2_hashmap_size = 19;
        args.ConvC = 32;
        args.ConvSz = 3;
        args.Output_size = [256,256];
    end

    layers = [
        featureInputLayer(2,"Name","xyz-input");
        network.Hash_EncodeingLayer( ...
            "base_res",     args.base_res,...
            "high_res",     args.high_res,...
            "device",       gpuDevice(),...
            "bounding_box", args.bounding_box,...
            "level",        args.hash_level,...
            "feature_len",  args.feature_dim,...
            "log2_hashmap_size", args.log2_hashmap_size);
    ];

    layers = [layers;
        network.padreshapeLayer("imgSize",args.Output_size)
        depthToSpace2dLayer([2,2],"Mode","CRD");
        convolution2dLayer(args.ConvSz,args.ConvC,"Stride",2,...
                                                  "Padding","same",...
                                                  "DilationFactor",1);
    ];
    
    conv2d = @(sz,ch) convolution2dLayer(sz,ch,"Stride",1,...
                                               "Padding","same",...
                                               "DilationFactor",1);

    for con = 1:args.layers
        layers = [layers;
                  reluLayer();
                  conv2d(args.ConvSz, args.ConvC);
                 ];
    end

    layers = [layers;
              reluLayer();
              conv2d(args.ConvSz, args.output_dims);
             ];

    
    net = dlnetwork(layers);
end