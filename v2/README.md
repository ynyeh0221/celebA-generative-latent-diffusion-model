# Class-Conditional VAE-Diffusion Model for Face Attribute Generation

This repository contains a PyTorch implementation of a class-conditional diffusion model for generating facial images with specific attributes, specifically targeting the "Smiling" attribute in the CelebA dataset.

## Overview

The project implements an advanced generative model that combines a Variational Autoencoder (VAE) with a conditional diffusion model to generate high-quality facial images with control over the smiling attribute. The model can generate both smiling and non-smiling faces with good quality and attribute control.

## Features

- **Enhanced VAE Architecture**: Includes channel attention, spatial attention, and uses perceptual loss for improved image quality
- **Class-Conditional Diffusion Model**: Operates in the latent space to generate realistic variations of faces
- **Comprehensive Visualization Tools**: Includes tools for visualizing reconstructions, latent space, denoising steps, and animations

## Model Architecture

### Variational Autoencoder (VAE)

The VAE consists of:
- An encoder that compresses input images into a latent space representation
- A decoder that reconstructs images from the latent representation
- Various improvements including:
  - Residual blocks with channel and spatial attention
  - Layer normalization and group normalization for stable training
  - Swish activation functions
  - VGG-based perceptual loss
  - GAN-like adversarial training with a discriminator

### Diffusion Model

The conditional diffusion model:
- Operates in the latent space of the trained VAE
- Uses a specialized UNet architecture adapted for latent vectors
- Incorporates time embeddings and class embeddings for conditioning
- Generates high-quality samples with attribute control

## Requirements

- PyTorch
- torchvision
- NumPy
- matplotlib
- tqdm
- scikit-learn
- imageio
- PIL

## Usage

### Dataset Preparation

The code expects the CelebA dataset with the following structure:
```
/content/celeba_data/
├── img_align_celeba/
│   └── [image files]
├── list_attr_celeba.txt
└── list_eval_partition.txt
```

### Training the VAE

```python
autoencoder = SimpleAutoencoder(in_channels=3, latent_dim=256, num_classes=2).to(device)
autoencoder, discriminator, ae_losses = train_autoencoder(
    autoencoder,
    train_loader,
    num_epochs=2000,
    lr=1e-4,
    lambda_cls=0.3,
    lambda_center=0.1,
    lambda_vgg=0.4,
    visualize_every=1,
    save_dir=results_dir
)
```

### Training the Diffusion Model

```python
conditional_unet = ConditionalUNet(
    latent_dim=256,
    time_emb_dim=256,
    num_classes=2,
    base_channels=64
).to(device)

conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
    autoencoder, 
    conditional_unet, 
    train_loader, 
    num_epochs=remaining_epochs, 
    lr=1e-3,
    visualize_every=5,
    save_dir=results_dir,
    device=device,
    start_epoch=start_epoch
)
```

### Sample Generation

```python
# Generate grid of samples
grid_path = generate_samples_grid(autoencoder, diffusion, n_per_class=5, save_dir=results_dir)

# Generate samples for a specific class
samples = generate_class_samples(autoencoder, diffusion, target_class="Smiling", num_samples=5)

# Create diffusion animation
animation_path = create_diffusion_animation(
    autoencoder, diffusion, class_idx=1,  # 1 for "Smiling"
    num_frames=50, fps=15,
    save_path="diffusion_animation.gif"
)
```

## Visualization Functions

The code includes several visualization functions:
- `visualize_reconstructions`: Displays original and reconstructed images
- `visualize_latent_space`: Projects latent space to 2D using t-SNE
- `visualize_denoising_steps`: Shows the denoising process from random noise to a generated image
- `create_diffusion_animation`: Creates a GIF animation of the diffusion process
- `generate_samples_grid`: Creates a grid of samples for different classes

## Model Components

The implementation includes:
- `SimpleAutoencoder`: The main VAE model
- `ConditionalUNet`: UNet architecture for the diffusion model
- `ConditionalDenoiseDiffusion`: Implements the diffusion process
- `CelebASmiling`: Custom dataset class for the CelebA dataset
- Various helper modules for attention, normalization, and activation

## Visualization Examples

The model can generate various visualizations:

1. **Sample Grid**: Examples of generated images from each class
2. **Latent Space Projection**: t-SNE or PCA visualization of encoded images
3. **Denoising Steps**: Progressive refinement from noise to final image
4. **Animated Diffusion Process**: GIF showing the full denoising sequence

 Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| GAN-VAE | ![Reconstructions](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v1/output/reconstruction/vae_reconstruction_epoch_75.png) | Original images (top) and their reconstructions (bottom) |
| Class Samples | ![Class Samples](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/diffusion_sample_result/sample_class_Not%20Smiling_epoch_180.png)![Class Samples](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/diffusion_sample_result/sample_class_Smiling_epoch_180.png) | Generated samples for cat and dog classes |
| Denoising Process | ![Denoising Not-Smiling](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/diffusion_path/denoising_path_Not%20Smiling_epoch_180.png)![Denoising Smiling](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/diffusion_path/denoising_path_Smiling_epoch_180.png)  | Visualization of generation process |
| Animation | ![Not-Smiling Animation](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/animination/diffusion_animation_Not%20Smiling_epoch_180.gif)![Smiling Animation](https://github.com/ynyeh0221/celebA-generative-latent-diffusion-model/blob/main/v2/output/animination/diffusion_animation_Smiling_epoch_180.gif) | Animation of the denoising process for generation |


## License

[MIT License](LICENSE)
