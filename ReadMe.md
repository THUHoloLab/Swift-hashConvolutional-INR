# SWAN: Swift hash-convolutional neural representation for differentiable computational imaging
  ![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?logo=apache)
  ![MATLAB](https://img.shields.io/badge/MATLAB-2024b-red.svg?logo=mathworks)
  ![NVIDIA GPU](https://img.shields.io/badge/gpu-nvidia-green?logo=nvidia)
  ![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg?logo=nvidia)
  
Authors: **[Shuhe Zhang](https://github.com/ShuheZhang-MUMC)**, and **[Liangcai Cao](https://scholar.google.com/citations?user=FYYb_-wAAAAJ&hl=en)**

:school: *[**HoloLab**](http://www.holoddd.com/), Tsinghua University*


<br>
This is the MATLAB code for the implementation of swift hash-convolutional neural representation (SWAN), a convolutional network-based neural representation for image combined with hash-based positional encoding. <br>
<br>
Inspired by the fast rendering properties of NVIDIA's Instant Neural Graphics Primitives ([Instant-NGP](https://github.com/NVlabs/instant-ngp)), SWAN employs a multi-resolution hash grid for positional encoding of input coordinates. The encoded feature vectors are then reshaped into a 2D feature map and decoded by a convolutional network to generate the final output image with desired properties. <br>
<br>

The hash grid and the decoding convolutional network are trained by minimizing the loss between the predicted image and the output image.
<div align = 'center'>
<img src = "https://github.com/THUHoloLab/Swift-hashConvolutional-INR/blob/main/resources/fig1.jpg" width = "800" alt="" align = center />
<br>
<em>SWAN for simple image representation</em>
</div>
<br>

📝 **Note 1**: The CUDA codes were writen and tested based on **CUDA v12.8** <br>
📝 **Note 2**: Ptychographic reconstruction requires **even-numbered** image dimensions for optimized fftshift operations. <br>
📝 **Note 3**: The code requires **MATLAB version at least 2024a** to ensure the [deep.DifferentiableFunction](https://www.mathworks.com/help/deeplearning/ug/define-custom-deep-learning-operations.html) class exist. <br>

## Key Features
- **Multi-Resolution Hash Encoding:** Efficient spatial encoding using hash tables at multiple grid resolutions, enabling compact representation of high-frequency details
- **Convolutional Decoding:** Reshapes encoded features into a 2D feature map that preserves spatial relationships, leveraging convolutional layers for natural image synthesis
- **End-to-End Training:** Optimizes both hash grid features and network weights simultaneously for a compact neural representation
- **Fast Inference:** Benefits from both the efficiency of hash-based encoding and the parallel processing capabilities of convolutional networks
<br>

## Why SWAN
- **Efficient hybrid design**: Combines Instant NGP's hash encoding with CNN's local modeling
- **Grid-based features**: Reshapes hash features into a 2D grid for convolutional processing
- **Single-pass generation**: Produces complete images in one forward pass, no point-by-point inference
- **Global operation friendly**: Natively supports FFT and other computational imaging operators

<div align="center">
  
| Method | Point-wise Inference | Global Operation Adaptation | Memory Efficiency | Image Prior |
| :--- | :--- | :--- | :--- | :--- |
| Pure MLP (SIREN) | ✅ Fully point-wise | ❌ Poor | High | Weak |
| Instant-NGP | ✅ Point-wise + Hash | ❌ Moderate | Extremely High | Weak |
| **SWAN** | ❌ Batch Processing | ✅ Good | High | Moderate (via Conv) |
| CNN Generator | ❌ Single-pass | ✅ Good | Low | Strong |
  
</div>
<br>

## How to use ?
### Fitting a single image using SWAN
For researchers interested in basic SWAN implementation, we provide **MATLAB codes** - a MATLAB adaptation of NVIDIA's Instant Neural Graphics Primitives ([Instant-NGP](https://github.com/NVlabs/instant-ngp)). 
This implementation enables efficient single image reconstruction through coordinate-based neural representations similar to a classicial neural representation for a single image.

**Implementation Details:**
The complete SWAN for single image fitting is available in the [MATLAB directory](https://github.com/THUHoloLab/Swift-hashConvolutional-INR/tree/main/fit_single_image), including example usage and comparison between MLP-based and NGP-based neural representations.
<div align = 'center'>
<img src = "https://github.com/THUHoloLab/Swift-hashConvolutional-INR/blob/main/resources/fig2.jpg" width = "800" alt="" align = center />
<br>
<em>Fitting ability of SWAN</em>
</div>
<br
