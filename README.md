# FatPy

[![Build Status](https://img.shields.io/github/actions/workflow/status/vybornak2/FatPy/python-ci.yml?label=Build)](https://github.com/vybornak2/FatPy/actions/workflows/python-ci.yml)
[![Documentation](https://img.shields.io/github/actions/workflow/status/vybornak2/FatPy/deploy_docs.yml?label=Documentation)](https://vybornak2.github.io/FatPy/)
[![Code Coverage](https://codecov.io/gh/vybornak2/FatPy/branch/main/graph/badge.svg)](https://codecov.io/gh/vybornak2/FatPy)
[![PyPI Version](https://img.shields.io/pypi/v/fatpy.svg?label=PyPI)](https://pypi.org/project/FatPy/)
[![Python Version](https://img.shields.io/pypi/pyversions/FatPy.svg?label=Python)](https://pypi.org/project/FatPy/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for fatigue life evaluation of materials.

## Features

- **Stress-Life** - Methods for analyzing fatigue life based on stress data
- **Strain-Life** - Tools for evaluating fatigue life using strain data
- **Energy-Life** - Methods for assessing fatigue life based on energy data

## Quick Links

- [Installation Guide](https://vybornak2.github.io/FatPy/development/install/)
- [Contributing Guide](CONTRIBUTING.md)
- [Documentation](https://vybornak2.github.io/FatPy/)
- [API Reference](https://vybornak2.github.io/FatPy/api/)
- [Theory Reference](https://vybornak2.github.io/FatPy/theory/)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## Installation

```bash
pip install fatpy
```

For development installation and more options, see our [detailed installation guide](https://vybornak2.github.io/FatPy/development/install/).

## Documentation

The documentation includes:

- [API Reference](https://vybornak2.github.io/FatPy/api/) - Detailed documentation of modules, classes, and functions
- [Theory Reference](https://vybornak2.github.io/FatPy/theory/) - Mathematical and physical background for implemented methods
- [Development Guides](https://vybornak2.github.io/FatPy/development/) - Information for contributors

### Building Documentation Locally

```bash
# Install development dependencies
pip install -e .

# Build and serve documentation
mkdocs serve
```

Visit `http://127.0.0.1:8000` to view the documentation locally.

## Contributing

We welcome contributions and follow Test-Driven Development (TDD) principles. Please see our [Contributing Guide](CONTRIBUTING.md) for:

- Setting up your development environment
- Test-Driven Development workflow
- Code quality guidelines and standards
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

Jan Vyborny - jan.vyborny2@gmail.com
Project Link: [github.com/vybornak2/fatpy](https://github.com/vybornak2/FatPy)
