classdef Hash_EncodeingLayer < nnet.layer.Layer & ...
                               nnet.layer.Formattable
    properties 
        device;
        level;
        base_res;
        high_res;
        log2_hashmap_size;
        hash_map_sizes;
    end

    properties(Access = private)
        log_scale;
        xxfunc;
        feature_len;
        total_param_size;
        offsets;
        bounding_box;
    end

    properties (Learnable)
        embedding;
    end

    methods 
        function self = Hash_EncodeingLayer(args)
            arguments
                args.device;
                args.bounding_box = [0,0;1,1];
                args.level = 8;
                args.base_res = 16;
                args.high_res = 4096;
                args.feature_len = 2;
                args.log2_hashmap_size = 21;
                args.name = "Hash-Encoder"
            end

            self.Name = args.name;
            self.device             = args.device;
            self.bounding_box       = gpuArray(single(args.bounding_box));
            self.level              = args.level;
            self.base_res           = args.base_res;
            self.high_res           = args.high_res;
            self.feature_len        = args.feature_len;
            self.log2_hashmap_size  = args.log2_hashmap_size;

            self.log_scale = log2(self.high_res/self.base_res) ...
                                 / (self.level - 1);

            self.NumInputs = 1;
            self.NumOutputs = 1;  

            self.hash_map_sizes = zeros(self.level,1);
            self.offsets = zeros(self.level,1);
            % self.foo = @(x) gpuArray(extractdata(x));
            self.xxfunc = network.fused_HashEncoding2D();
        end

        function self = initialize(self,layout)
            % Init learnable parameters
            if isempty(self.embedding) 
                
                offset = 0;
                for levels = 1:self.level
                    resolution = self.base_res * 2^((levels-1) * ...
                                                     self.log_scale) - 1;
                    resolution = ceil(resolution) + 1;
                    full_size_aligned = ceil(resolution^2 / 32) * 32;
                    params_size_level = min(2^self.log2_hashmap_size, ...
                                              full_size_aligned);
                    % params_size_level = 2^self.log2_hashmap_size;
                    self.hash_map_sizes(levels) = params_size_level;
                    self.offsets(levels) = offset;
                    offset = offset + params_size_level;
                end

                self.total_param_size = offset * self.feature_len;

                self.embedding = 0.002 * gpuArray.rand(self.feature_len, ...
                                                       offset, ...
                                                       'single') - 0.001;

                self.hash_map_sizes = gpuArray(single(self.hash_map_sizes));
                self.offsets = gpuArray(single(self.offsets));
            end
        end

        function Y = predict(self,X) 
            X_dim = dims(X);
            
            if ~isgpuarray(X)
                X = gpuArray(X);
            end

            if ~isgpuarray(self.embedding)
                self.embedding = gpuArray(self.embedding);
            end

            Y = self.xxfunc( ...
                X,...
                self.embedding,...
                ... % input parameters for CUDA kernel
                self.bounding_box,...
                self.hash_map_sizes,...
                self.offsets,...
                self.level,...
                self.log_scale,...
                self.base_res,...
                self.feature_len...
            );

            Y = dlarray(Y, X_dim);
        end
    end
end