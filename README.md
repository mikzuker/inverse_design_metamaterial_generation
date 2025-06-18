# Metasurface Optimization for Spectrum Matching

This project implements an optimization framework for designing metasurfaces that can mimic specific scattering patterns. The optimization is performed using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.

## Project Structure

```
sphere_metasurface/
├── optimization.py      # Main optimization implementation using CMA-ES
├── parametrization.py   # Surface parametrization and geometry generation
├── fitness_function.py  # Loss function and spectrum calculation
├── dataset_creation.py  # Dataset generation for training
└── experiment_reproducing.py  # Reproduce optimization results
```

## Features

- Optimization of sphere-based metasurfaces using CMA-ES
- Support for both single and multiple target angles
- Parallel computation support using Ray
- Visualization of optimization results
- Reproducible experiments with saved parameters

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- CMA-ES
- Ray (for parallel processing)
- Smuthi (for electromagnetic simulations)

## Usage

### Basic Optimization

```python
from optimization import Optimization

# Initialize optimization
opt = Optimization(
    number_of_cells=3,
    side_length=9,
    refractive_index=1.5,
    vacuum_wavelength=1,
    object_to_mimic=target_object,
    angeles_to_mimic=[30, 75, 140],
    seed=42
)

# Run optimization
opt.optimize()
```

## Optimization Parameters

- `number_of_cells`: Grid size for the metasurface (e.g., 3x3)
- `side_length`: Physical size of the metasurface
- `refractive_index`: Refractive index of the surface material
- `vacuum_wavelength`: Wavelength of the incident light
- `object_to_mimic`: Target object whose scattering pattern to mimic
- `angeles_to_mimic`: List of angles at which to match the scattering pattern
- `seed`: Random seed for reproducibility

## Output

The optimization process generates:
- Optimization progress plots
- Final surface geometry visualization
- Scattering pattern comparison plots
- Saved parameters for reproducibility

## License

MIT License.