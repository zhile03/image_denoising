# DnCNN-S Image Denoising Project

This project reproduces the image denoising model **DnCNN-S** as described in the paper:  
**"Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"**.

## Training Details

- **Dataset**: The model is trained on CIFAR-10 or BSDS500 dataset, with image patches extracted for training.
- **Model Configuration**:  
  - The model is trained for 50 epochs.
  - Learning rate starts at 0.1 and exponentially decays to 0.0001.
  - Stochastic Gradient Descent (SGD) optimizer with momentum = 0.9 and weight decay = 0.0001.
- **Evaluation**: The model is evaluated on BSD68 and Set12 datasets using PSNR and SSIM metrics.
