# CelebA Conditional Diffusion Model

A sophisticated deep learning implementation that combines Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), and Diffusion Models to generate high-quality face images conditional on the "Smiling" attribute from the CelebA dataset.

## Overview

This project implements an advanced generative modeling pipeline that:

1. Trains a VAE-GAN hybrid model for efficient image encoding/decoding
2. Builds a conditional diffusion model in the latent space
3. Generates high-quality face images controlled by attribute conditions
4. Provides extensive visualization tools to understand the generative process

## Features

- **Hybrid VAE-GAN Architecture**: Combines reconstruction quality of VAEs with the sharpness of GANs
- **Conditional Diffusion Model**: Generates images based on specified attributes (smiling/not smiling)
- **Advanced Training Techniques**:
  - Perceptual loss using VGG features
  - Center loss for better feature clustering
  - Multi-stage training strategy with warm-up phases
- **Rich Visualizations**:
  - Latent space projections (t-SNE)
  - Denoising process animations
  - Class-conditional sample generation
  - Reconstruction quality assessment

## Requirements

- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn
- tqdm
- imageio

The script is designed to run on Google Colab with access to GPU acceleration.

## Dataset

The model uses the CelebA dataset, specifically focusing on the "Smiling" attribute:
- 64x64 face images
- Binary classification (Smiling vs Not Smiling)
- Training images are center-cropped to focus on facial features

## Model Architecture

### Variational Autoencoder
- **Encoder**: Convolutional network with residual blocks and attention mechanisms
- **Latent Space**: 256-dimensional space with learned class centers
- **Decoder**: Transposed convolutions with residual connections

### GAN Component
- **Discriminator**: 4-layer convolutional network for adversarial training
- **Adversarial Loss**: Combined with reconstruction losses for better image quality

### Diffusion Model
- **UNet**: Conditional architecture with time and class embeddings
- **Sampling Process**: Iterative denoising with 1000 diffusion steps
- **Conditioning**: Class labels guide the generation process

## Usage

```python
# Main training workflow
main(total_epochs=10000)

# Generate samples for a specific class
generate_class_samples(autoencoder, diffusion, target_class="Smiling", num_samples=5)

# Create denoising animation
create_diffusion_animation(autoencoder, diffusion, class_idx=1, num_frames=50, fps=15)
```

## Visualization Examples

The model can generate various visualizations:

1. **Sample Grid**: Examples of generated images from each class
2. **Latent Space Projection**: t-SNE or PCA visualization of encoded images
3. **Denoising Steps**: Progressive refinement from noise to final image
4. **Animated Diffusion Process**: GIF showing the full denoising sequence

 Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| GAN-VAE | ![Reconstructions](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v1/output/reconstruction/vae_reconstruction_epoch_75.png) | Original images (top) and their reconstructions (bottom) |
| Class Samples | ![Class Samples](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v1/output/diffusion_sample_result/sample_class_Not%20Smiling_epoch_150.png)![Class Samples](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v1/output/diffusion_sample_result/sample_class_Smiling_epoch_150.png) | Generated samples for cat and dog classes |
| Denoising Process | ![Denoising Not-Smiling](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v1/output/diffusion_path/denoising_path_Not%20Smiling_epoch_150.png)![Denoising Smiling](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v1/output/diffusion_path/denoising_path_Smiling_epoch_150.png)  | Visualization of generation process |
| Animation | ![Not-Smiling Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v10/output/diffusion_animination/diffusion_animation_cat_epoch_800.gif)![Smiling Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v10/output/diffusion_animination/diffusion_animation_dog_epoch_800.gif) | Animation of the denoising process for generation |

## Training Process

The training happens in two main phases:

1. **VAE-GAN Training**:
   - Initial reconstruction-focused phase
   - Gradual introduction of KL divergence
   - Later integration of classification and center losses
   - Final GAN training for image quality enhancement

2. **Diffusion Model Training**:
   - Operates in the VAE's latent space
   - Requires pre-trained, frozen VAE
   - Uses conditional sampling based on class labels

## Saving and Loading

Models are automatically saved during training:
- Checkpoints at regular intervals
- Final models saved at completion
- Visualizations saved to the specified results directory

## Credits

This implementation uses several advanced techniques from recent research in generative modeling, including:
- Attention mechanisms in VAEs
- Perceptual losses for image quality
- Latent diffusion modeling
