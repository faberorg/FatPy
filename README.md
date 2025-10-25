# FatPy

[![Build Status](https://img.shields.io/github/actions/workflow/status/faberorg/FatPy/python-ci.yml?label=Build)](https://github.com/faberorg/FatPy/actions/workflows/python-ci.yml)
[![Documentation](https://img.shields.io/github/actions/workflow/status/faberorg/FatPy/deploy_docs.yml?label=Documentation)](https://faberorg.github.io/FatPy/)
[![Code Coverage](https://codecov.io/gh/faberorg/FatPy/branch/main/graph/badge.svg)](https://codecov.io/gh/faberorg/FatPy)
[![PyPI Version](https://img.shields.io/pypi/v/fatpy.svg?label=PyPI)](https://pypi.org/project/FatPy/)
[![Python Version](https://img.shields.io/pypi/pyversions/FatPy.svg?label=Python)](https://pypi.org/project/FatPy/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for fatigue life evaluation of materials.

## Features

### Core Modules

- **Stress-Life** - Stress-based fatigue assessment methods for estimating fatigue damage with corrections for various effects (stress concentration, size, surface quality, etc.)
- **Strain-Life** - Fatigue analysis using strain amplitude and cycles to failure relationships (ε-N approaches) such as Coffin-Manson and Basquin laws, suited for low-cycle and transitional fatigue regimes
- **Energy-Life** - Fatigue analysis methods based on the relationship between strain energy density and number of cycles to failure
- **Damage Cumulation** - Various fatigue damage accumulation rules for variable amplitude loading, including linear models (Palmgren-Miner) and advanced non-linear approaches accounting for load sequence effects
- **Decompositions** - Methods for breaking down complex load signals into cycles, containing both uniaxial and multiaxial procedures
- **Plane-Based Methods** - Methods for processing stress tensor paths on material planes, providing infrastructure for critical-plane and integral prediction approaches

### Supporting Modules

- **Material Laws** - Functions dealing with various material constitutive laws and behavior models
- **Structural Mechanics** - Stress, strain analysis and transformation utilities
- **Utils** - General utilities for mesh handling, signal processing, and data manipulation

## Quick Links

- [Installation Guide](https://faberorg.github.io/FatPy/development/install/)
- [Contributing Guide](CONTRIBUTING.md)
- [Documentation](https://faberorg.github.io/FatPy/)
- [API Reference](https://faberorg.github.io/FatPy/api/)
- [Theory Reference](https://faberorg.github.io/FatPy/theory/)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## Installation

```bash
pip install fatpy
```

For development installation and more options, see our [detailed installation guide](https://faberorg.github.io/FatPy/development/install/).

## Documentation

The documentation includes:

- [API Reference](https://faberorg.github.io/FatPy/api/) - Detailed documentation of modules, classes, and functions
- [Theory Reference](https://faberorg.github.io/FatPy/theory/) - Mathematical and physical background for implemented methods
- [Development Guides](https://faberorg.github.io/FatPy/development/) - Information for contributors

### Building Documentation Locally

```bash
# Install development dependencies
pip install -e .

# Build and serve documentation
mkdocs serve
```

Visit `http://127.0.0.1:8000` to view the documentation locally.

## Contributing

We welcome contributions. Please see our [Contributing Guide](CONTRIBUTING.md) for:

- Setting up your development environment
- Code quality guidelines and standards
- Testing procedures
- Documentation requirement
- Pull request process

## Testing

FatPy uses pytest for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/fatpy
```

## The FABER Project

FatPy is a key initiative of Working Group 6 (WG6) within the [FABER](https://faber-cost.eu/) (Fatigue Benchmark Repository) project, operating under COST Action CA23109.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contact

Jan Vyborny - <jan.vyborny2@gmail.com>
Project Link: [github.com/faberorg/fatpy](https://github.com/faberorg/FatPy)
