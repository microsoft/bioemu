# BioEMU Hydra Run

This directory contains `hydra_run.py`, an alternative entry point for BioEMU that uses Hydra configuration management instead of the CLI interface.

## Overview

`hydra_run.py` provides the same functionality as the CLI interface but uses YAML configuration files for easier experimentation and parameter management. It maintains full compatibility with the existing BioEMU sampling pipeline while offering the benefits of Hydra's configuration system.

## Usage

### Basic Usage
```bash
# Run with default configuration
python hydra_run.py

# Override specific parameters
python hydra_run.py sequence=GYDPETGTWG num_samples=64

# Override steering parameters
python hydra_run.py steering.num_particles=3 steering.start=0.3
```

### Configuration Files

The script uses the following configuration hierarchy:
- **Main config**: `../src/bioemu/config/bioemu.yaml` (includes steering control parameters)
- **Denoiser config**: `../src/bioemu/config/denoiser/dpm.yaml` (default)
- **Potentials config**: `../src/bioemu/config/steering/physical_potentials.yaml` (referenced by main config)

### Key Features

1. **Hydra Integration**: Full Hydra configuration management with overrides
2. **Steering Support**: Complete steering configuration with physical potentials
3. **Reproducibility**: Fixed seeds for consistent results
4. **Error Handling**: Comprehensive error reporting and logging
5. **Output Management**: Organized output directories with descriptive names

### Configuration Parameters

#### Basic Parameters
- `sequence`: Amino acid sequence to sample
- `num_samples`: Number of samples to generate
- `batch_size_100`: Batch size for sequences of length 100

#### Steering Parameters
- `steering.num_particles`: Number of particles per sample (1 = no steering)
- `steering.start`: Start time for steering (0.0-1.0)
- `steering.end`: End time for steering (0.0-1.0)
- `steering.resampling_freq`: Resampling frequency
- `steering.fast_steering`: Enable fast steering mode
- `steering.potentials`: Reference to potentials config file (e.g., "physical_potentials")

#### Denoiser Parameters
- `denoiser.N`: Number of denoising steps
- `denoiser.eps_t`: Final timestep
- `denoiser.max_t`: Initial timestep
- `denoiser.noise`: Noise level

### Examples

#### Basic Sampling
```bash
python hydra_run.py sequence=GYDPETGTWG num_samples=10
```

#### Steering with Custom Parameters
```bash
python hydra_run.py \
    sequence=GYDPETGTWG \
    num_samples=20 \
    steering.num_particles=3 \
    steering.start=0.3 \
    steering.fast_steering=true
```

#### High-Throughput Sampling
```bash
python hydra_run.py \
    sequence=MTEIAQKLKESNEPILYLAERYGFESQQTLTRTFKNYFDVPPHKYRMTNMQGESRFLHPL \
    num_samples=128 \
    batch_size_100=800
```

### Output

The script generates:
- **PDB files**: Individual structure files
- **XTC files**: Trajectory files for visualization
- **NPZ files**: Raw tensor data
- **Logs**: Detailed sampling information

Output is organized in `./outputs/hydra_run/` with descriptive directory names.

### Comparison with CLI Interface

| Feature | CLI Interface | Hydra Run |
|---------|---------------|-----------|
| Configuration | Command-line arguments | YAML files + overrides |
| Reproducibility | Manual seed setting | Automatic fixed seeds |
| Experimentation | Manual parameter changes | Easy YAML overrides |
| Documentation | Help text | Inline YAML comments |
| Version Control | Command history | Configuration files |

### Troubleshooting

1. **Import Errors**: Ensure you're running from the `notebooks/` directory
2. **CUDA Issues**: The script automatically detects and uses available GPUs
3. **Memory Issues**: Reduce `batch_size_100` for longer sequences
4. **Configuration Errors**: Check YAML syntax and file paths

### Development

To modify the configuration:
1. Edit the relevant YAML files in `../src/bioemu/config/`
2. Test with small parameter changes
3. Use Hydra's `--help` flag to see available options

For more advanced usage, refer to the [Hydra documentation](https://hydra.cc/).
