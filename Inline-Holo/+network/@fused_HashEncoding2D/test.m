clc
clear

% addpath('private');

data = [];
level = 8;
base_res = 16;
high_res = 4096;
feature_len = 2;
log_scale = log(high_res/base_res) ...
                                 / (level - 1);

bounding_box = gpuArray(single([0,0;1,1]));

log2_hashmap_size = 19;

hash_offsets = zeros(level,1);
hash_map_sizes = zeros(level,1);

offset = 0;
for levels = 1:level
    resolution = base_res * exp(levels * log_scale) - 1;
    resolution = ceil(resolution) + 1;
    full_size = resolution^2;
    full_size_aligned = ceil( (full_size + 4 - 1) / 4 );
    params_size_level = min(2^log2_hashmap_size, full_size_aligned);
    hash_map_sizes(levels) = params_size_level;
    hash_offsets(levels) = offset;
    offset = offset + params_size_level;
end

total_param_size = offset * feature_len;
embedding = 0.002 * rand(feature_len, offset, 'single') - 0.001;
embedding = gpuArray(embedding);

xyz = gpuArray(single(rand(2,16)));

hash_map_sizes = uint32(gpuArray(single(hash_map_sizes)));
hash_offsets = uint32(gpuArray(single(hash_offsets)));

out_feature = fullyfused_hashEncoder( ...
    'forward',...
    xyz,...
    embedding,...
    bounding_box,...
    hash_map_sizes,...
    hash_offsets,...
    level,...
    log_scale,...
    base_res,...
    feature_len ...
);