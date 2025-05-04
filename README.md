# Class-Conditional VAE-Diffusion Model for Face Attribute Generation

This repository contains a PyTorch implementation of a class-conditional diffusion model for generating facial images with specific attributes, specifically targeting the "Smiling" attribute in the CelebA dataset.

## Overview

The project implements an advanced generative model that combines a Variational Autoencoder (VAE) with a conditional diffusion model to generate high-quality facial images with control over the smiling attribute. The model can generate both smiling and non-smiling faces with good quality and attribute control.

[DEEPWIKI](https://deepwiki.com/ynyeh0221/celebA-generative-latent-diffusion-model)

## Features

- **Enhanced VAE Architecture**: Includes channel attention, spatial attention, and uses perceptual loss for improved image quality
- **Class-Conditional Diffusion Model**: Operates in the latent space to generate realistic variations of faces
- **Comprehensive Visualization Tools**: Includes tools for visualizing reconstructions, latent space, denoising steps, and animations

## Visualization Examples

The model can generate various visualizations:

1. **Sample Grid**: Examples of generated images from each class
2. **Latent Space Projection**: t-SNE or PCA visualization of encoded images
3. **Denoising Steps**: Progressive refinement from noise to final image
4. **Animated Diffusion Process**: GIF showing the full denoising sequence

 Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| GAN-VAE (75 epoch) | ![Reconstructions](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v1/output/reconstruction/vae_reconstruction_epoch_75.png) | Original images (top) and their reconstructions (bottom) |
| Class Samples (180 epoch) | ![Class Samples](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/diffusion_sample_result/sample_class_Not%20Smiling_epoch_180.png)![Class Samples](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/diffusion_sample_result/sample_class_Smiling_epoch_180.png) | Generated samples for cat and dog classes |
| Denoising Process (180 epoch) | ![Denoising Not-Smiling](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/diffusion_path/denoising_path_Not%20Smiling_epoch_180.png)![Denoising Smiling](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/diffusion_path/denoising_path_Smiling_epoch_180.png)  | Visualization of generation process |
| Animation (180 epoch) | ![Not-Smiling Animation](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/animination/diffusion_animation_Not%20Smiling_epoch_180.gif)![Smiling Animation](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/animination/diffusion_animation_Smiling_epoch_180.gif) | Animation of the denoising process for generation |
