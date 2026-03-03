function net = NGP(args)
    arguments
        args.output_dims = 2;
        args.layers = 2;
        args.hash_level = 8;
        args.base_res = 16;
        args.high_res = 4096;
        args.bounding_box = [0,0;1,1];
        args.feature_dim = 2;
        args.log2_hashmap_size = 19;
        args.MLP_width = 32;
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

    if args.layers > 0
        for con = 1:args.layers
            layers = [layers;
                network.FC_SimpleLayer(args.MLP_width,"Name","fc" + con);
                reluLayer();
            ];
        end

        layers = [layers;
            network.FC_SimpleLayer(args.output_dims,"Name",'fin');
        ];
    end

    net = dlnetwork(layers);
end