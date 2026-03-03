<!-- TOC --><a name="Swift hash-convolutional neural representation"></a>
# SWAN: Swift hash-convolutional neural representation for differentiable computational imaging
  ![License](https://img.shields.io/badge/License-MIT-blue.svg?logo=apache)
  ![MATLAB](https://img.shields.io/badge/MATLAB-2024b-red.svg?logo=mathworks)
  ![NVIDIA GPU](https://img.shields.io/badge/gpu-nvidia-green?logo=nvidia)
  ![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg?logo=nvidia)
  
Authors: **[Shuhe Zhang](https://github.com/ShuheZhang-MUMC)**, and **[Liangcai Cao](https://scholar.google.com/citations?user=FYYb_-wAAAAJ&hl=en)**

:school: *[**HoloLab**](http://www.holoddd.com/), Tsinghua University*


<br>
This is the MATLAB code for the implementation of swift hash-convolutional neural representation (SWAN), a convolutional network-based neural representation for image combined with hash-based positional encoding. <br>
<br>
Inspired by the fast rendering properties of Instant Neural Graphics Primitives (Instant-NGP), SWAN employs a multi-resolution hash grid for positional encoding of input coordinates. The encoded feature vectors are then reshaped into a 2D feature map and decoded by a convolutional network to generate the final output image with desired properties. <br>
<br>

📝 **Note 1**: The CUDA codes were writen and tested based on **CUDA v12.8**
📝 **Note 2**: Ptychographic reconstruction requires **even-numbered** image dimensions to facilitate optimized FFT shift operations. <br>
📝 **Note 3**: The code requires **MATLAB version at least 2024a** to ensure the [deep.DifferentiableFunction](https://www.mathworks.com/help/deeplearning/ug/define-custom-deep-learning-operations.html) class exist.

## Key Features
- **Multi-Resolution Hash Encoding:** Efficient spatial encoding using hash tables at multiple grid resolutions, enabling compact representation of high-frequency details
- **Convolutional Decoding:** Reshapes encoded features into a 2D feature map that preserves spatial relationships, leveraging convolutional layers for natural image synthesis
- **End-to-End Training:** Optimizes both hash grid features and network weights simultaneously for a compact neural representation
- **Fast Inference:** Benefits from both the efficiency of hash-based encoding and the parallel processing capabilities of convolutional networks
<br>
