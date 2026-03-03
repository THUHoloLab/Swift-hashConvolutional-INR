function net = NGP(args)
    arguments
        args.output_dims = 1;
        args.layers = 2;
        args.hash_level = 8;
        args.base_res = 16;
        args.high_res = 4096;
        args.bounding_box
        args.feature_dim = 2;
        args.log2_hashmap_size = 19;
        args.mlp_width = 64;
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


    for con = 1:args.layers
        layers = [layers;
            fullyConnectedLayer(args.mlp_width);
            reluLayer();
        ];
    end

    if args.layers > 0
    layers = [layers;
        fullyConnectedLayer(args.output_dims);
    ];
    end

    net = dlnetwork(layers);
end