# Diffusion Model for Electromagnetic Inverse Design Problem

This repository contains implementations of both **conditional** and **unconditional** diffusion models for 1D data generation, built with PyTorch. The models are designed to solve inverse design scattering problems. The outputs of the models are vectors with relative values which can then be decoded to generate the geometry of metamaterials made of dielectric spheres. The project also includes comprehensive parametrization tools and CMA-ES optimization capabilities for geometry optimization based on target spectra.

## 🚀 Features

- **Unconditional Diffusion Model**: Generates 1D sequences without any conditioning
- **Conditional Diffusion Model**: Generates 1D sequences conditioned with FiLM on specific parameters (DSCS values on specified angles)
- **1D UNet Architecture**: Optimized for sequential data processing
- **Multiple Sampling Methods**: Standard sampling and DDIM sampling for faster generation
- **Training Visualization**: Tools for analyzing training loss and denoising process
- **Dataset Utilities**: Comprehensive tools for dataset preparation and analysis
- **Metasurface Parametrization**: Complete framework for dielectric sphere-based metamaterial design
- **CMA-ES Optimization**: Covariance Matrix Adaptation Evolution Strategy for geometry optimization
- **Electromagnetic Simulation**: Integration with SMUTHI for accurate scattering calculations
- **Target Spectrum Matching**: Optimization towards desired electromagnetic response

## 📁 Project Structure

```
diffusion_model/                   # Diffusion model implementations
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

sphere_metasurface/               # Metasurface parametrization and optimization
├── parametrization.py            # Core parametrization class for sphere-based metasurfaces
├── optimization.py               # CMA-ES optimization framework
├── fitness_function.py           # Fitness functions for spectrum matching
├── mat_decoder.py                # Matrix decoder utilities
├── object_checker.py             # Object validation and checking
├── experiment_reproducing.py     # Experiment reproduction utilities
├── results/                      # Optimization results and outputs
├── plots/                        # Generated plots and visualizations
├── real_object_results/          # Real object simulation results
└── reproduced_experiments/       # Reproduced experimental data
```

## 🏗️ Metasurface Parametrization

The project includes a comprehensive framework for parametrizing and optimizing dielectric sphere-based metasurfaces. This allows you to convert the output vectors from diffusion models into actual physical geometries.

### Sphere Surface Parametrization

The `Sphere_surface` class provides a flexible way to create metasurfaces:

```python
from sphere_metasurface.parametrization import Sphere_surface

# Create a metasurface with 4x4 grid of cells
surface = Sphere_surface(
    number_of_cells=4,
    side_length=4.0,  # micrometers
    reflective_index=complex(2.0, 0.0)  # refractive index
)

# Generate the mesh
surface.mesh_generation()

# Add spheres with specific radii and positions
radii = [0.2, 0.3, 0.25, 0.35, ...]  # 16 values for 4x4 grid
positions = [[0.5, 0.5, 0], [1.5, 0.5, 0], ...]  # 16 positions

surface.__spheres_add__(radii, positions)

# Visualize the metasurface
surface.spheres_plot("metasurface.png")
```

### Key Features

- **Flexible Grid System**: Configurable number of cells and side length
- **SMUTHI Integration**: Direct integration with electromagnetic simulation software
- **2D Visualization**: Built-in plotting capabilities for metasurface geometry
- **Parameter Control**: Independent control over sphere radii and positions

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

The conditional model generates 1D sequences based on conditional vectors (DSCS values on specified angles):

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
```

## 🎯 CMA-ES Optimization Framework

The project implements Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for optimizing metasurface geometries to match target electromagnetic spectra.

### Optimization Setup

```python
from sphere_metasurface.optimization import Optimization
import smuthi.particles

# Define target object (e.g., a sphere to mimic)
target_sphere = smuthi.particles.Sphere(
    position=[0, 0, 0],
    refractive_index=complex(1.5, 0.0),
    radius=1.0
)

# Initialize optimization
optimizer = Optimization(
    object_to_mimic=target_sphere,
    vacuum_wavelength=633.0,  # nanometers
    angeles_to_mimic=[0, 30, 60, 90, 120, 150, 180],  # degrees
    side_length=4.0,  # micrometers
    number_of_cells=4,
    refractive_index=complex(2.0, 0.0),
    iterations=1000,
    seed=42,
    num_workers=4,  # Parallel processing
    use_parallel=True
)

# Run optimization
optimizer.optimize()
```

### Fitness Function

The optimization uses a custom fitness function that compares the scattering cross-section of the metasurface with the target object:

```python
from sphere_metasurface.fitness_function import calculate_loss

# Calculate loss between metasurface and target object
loss = calculate_loss(
    spheres_surface=surface,
    object=target_sphere,
    vacuum_wavelength=633.0,
    angles_to_mimic=[0, 30, 60, 90, 120, 150, 180],
    polar_angle=np.pi,  # from top
    azimuthal_angle=0,  # x-axis
    polarization=0  # TE polarization
)
```

### Optimization Features

- **Parallel Processing**: Ray-based parallelization for faster optimization
- **Adaptive Strategy**: CMA-ES automatically adapts to the optimization landscape
- **Spectrum Matching**: Optimizes towards specific target electromagnetic responses
- **Multiple Angles**: Simultaneous optimization across multiple scattering angles
- **Precomputation**: Efficient precomputation of target object spectra
