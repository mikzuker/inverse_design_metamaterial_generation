# Diffusion Model for Electromagnetic Inverse Design Problem

This repository contains implementations of both **conditional** and **unconditional** diffusion models for 1D data generation, built with PyTorch. The models are designed to solve inverse disign scattering problem. Outcomes of the models are vectors with relative values which then can be decoded to a geometry of the metamaterial made of dielectrical spheres (parametrization can be found in the parametrization folder). 

## 🚀 Features

- **Unconditional Diffusion Model**: Generates 1D sequences without any conditioning
- **Conditional Diffusion Model**: Generates 1D sequences conditioned on specific parameters (e.g., angles)
- **1D UNet Architecture**: Optimized for sequential data processing
- **Multiple Sampling Methods**: Standard sampling and DDIM sampling for faster generation
- **Training Visualization**: Tools for analyzing training loss and denoising process
- **Dataset Utilities**: Comprehensive tools for dataset preparation and analysis

## 📁 Project Structure

```
diffusion_model/
├── conditional_network/           # Conditional diffusion model implementation
│   ├── conditional_model.py      # Main conditional model class
│   ├── conditional_decoder.py    # Decoder for conditional generation
│   ├── conditional_diffusion_pytorch_1d.py  # Core conditional diffusion implementation
│   └── conditional_dataset_preparation.py   # Dataset preparation for conditional model
├── unconditional_network/        # Unconditional diffusion model implementation
│   ├── model.py                  # Main unconditional model class
│   ├── decoder.py                # Decoder for unconditional generation
│   ├── denoising_diffusion_pytorch_1d.py   # Core unconditional diffusion implementation
│   └── dataset_preparation.py    # Dataset preparation for unconditional model
├── conditional_csv_datasets/     # Pre-processed conditional datasets
├── Conditional_Model_16_2_4e-6_11000_20000/  # Trained conditional model checkpoints
├── dataset_csv_utils.py          # Utilities for loading datasets from CSV
├── dataset_creation.py           # Dataset creation utilities
├── mpe_analysis.py               # Mean Percentage Error analysis
├── mean_median.py                # Statistical analysis tools
├── plot_loss.py                  # Training loss visualization
├── dataset_pca_analysis.py       # PCA analysis for datasets
└── mpe_loss_analysis.png         # Analysis visualization
```

## 🎯 Usage

### Unconditional Diffusion Model

The unconditional model generates 1D sequences without any conditioning:

```python
from diffusion_model.unconditional_network.model import Diffusion_model
from pathlib import Path

# Initialize the model
model = Diffusion_model(milestone=1)

# Create dataset
dataset = model.__dataset_create__(
    Path('diffusion_model/training_dataset'), 
    angles=[0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 
            70, 80, 90, 100, 110, 120, 130, 140, 150]
)

# Train the model
model.__train__(
    dataset=dataset,
    train_batch_size=1,
    train_lr=8e-4,
    train_num_steps=10,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=True,
    loss_path=Path('diffusion_model/training_loss')
)

# Generate samples
sampled_seq = model.__sample__(batch_size=1)
print(sampled_seq.shape)  # torch.Size([1, 1, 32])
```

### Conditional Diffusion Model

The conditional model generates 1D sequences based on conditional vectors (e.g., angles):

```python
from diffusion_model.conditional_network.conditional_model import Diffusion_model
import torch

# Initialize the conditional model
model = Diffusion_model(
    milestone=1,
    use_film=True,
    cond_dim=10,
    train_batch_size=16,
    train_lr=4e-4,
    train_num_steps=2000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    timesteps=1000,
    amp=True,
    angles=[0, 10, 20, 40, 60, 80, 100, 120, 140, 160]
)

# Train the model
model.__train__()

# Generate conditional samples
conditional_vec = torch.randn(1, 10)  # 10-dimensional condition vector
sampled_seq = model.__sample__(batch_size=1, conditional_vec=conditional_vec)

# Use DDIM sampling for faster generation
sampled_seq_ddim = model.__sample_ddim__(
    shape=(1, 1, 16), 
    conditional_vec=conditional_vec
)

# Visualize the denoising process
denoising_steps = model.__visualize_denoising_process__(
    shape=(1, 1, 16), 
    conditional_vec=conditional_vec,
    specific_steps=[0, 100, 500, 1000]
)
```

## 🔧 Model Architecture

### UNet1D Architecture

Both models use a 1D UNet architecture with the following characteristics:

- **Unconditional Model**: 32-dimensional features, sequence length 32
- **Conditional Model**: 16-dimensional features, sequence length 16, with FiLM conditioning
- **Multi-scale Processing**: Uses dimension multipliers (1, 2, 4, 8) for hierarchical feature extraction
- **Residual Connections**: Enhanced with residual blocks for better gradient flow

### Diffusion Process

- **Timesteps**: Configurable number of diffusion steps (default: 1000)
- **Objective**: Uses `pred_v` objective for improved training stability
- **Sampling**: Supports both standard sampling and DDIM sampling
- **Conditioning**: FiLM-based conditioning for the conditional model

## 📊 Training and Analysis

### Training Configuration

Key training parameters:

- **Batch Size**: Configurable batch size for training
- **Learning Rate**: Adjustable learning rate with default 4e-4
- **Gradient Accumulation**: Support for gradient accumulation to handle large effective batch sizes
- **EMA Decay**: Exponential Moving Average for model weights (default: 0.995)
- **Mixed Precision**: Automatic Mixed Precision (AMP) support for faster training

### Analysis Tools

The project includes several analysis utilities:

- **Loss Analysis**: `plot_loss.py` for training loss visualization
- **Statistical Analysis**: `mean_median.py` for data statistics
- **PCA Analysis**: `dataset_pca_analysis.py` for dimensionality reduction analysis
- **MPE Analysis**: `mpe_analysis.py` for Mean Percentage Error calculations

## 📈 Performance

### Model Specifications

| Model Type | Sequence Length | Feature Dimensions | Conditioning |
|------------|----------------|-------------------|--------------|
| Unconditional | 32 | 32 | None |
| Conditional | 16 | 16 | FiLM (10D) |

### Training Recommendations

- **Unconditional Model**: Use for general sequence generation tasks
- **Conditional Model**: Use when you need to control generation based on specific parameters
- **Batch Size**: Start with smaller batch sizes and increase based on available memory
- **Learning Rate**: Begin with 4e-4 and adjust based on training stability

## 🚀 Advanced Features

### DDIM Sampling

For faster generation, use DDIM sampling:

```python
# Conditional model with DDIM sampling
sampled_seq = model.__sample_ddim__(shape=(1, 1, 16), conditional_vec=conditional_vec)
```

### Denoising Process Visualization

Visualize the step-by-step denoising process:

```python
denoising_steps = model.__visualize_denoising_process__(
    shape=(1, 1, 16), 
    conditional_vec=conditional_vec,
    specific_steps=[0, 100, 500, 1000]
)
