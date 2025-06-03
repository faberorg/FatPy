# Contributing to FatPy

Thank you for considering contributing to FatPy! This document provides guidelines and instructions for contributing to the project.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/FatPy.git`
3. Create a branch: `git checkout -b my-feature-branch`
4. Write tests for your feature (following TDD principles)
5. Implement the feature
6. Run tests and make sure they pass
7. Submit a pull request

## Development Setup

For detailed setup instructions, see the [Installation Guide](https://vybornak2.github.io/FatPy/development/install/).

### Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/your-username/FatPy.git
cd FatPy

# Using uv (recommended)
uv venv
.venv\Scripts\activate  # On Unix: source .venv/bin/activate
uv sync
uv pip install -e .
pre-commit install

# Using pip
python -m venv venv
venv\Scripts\activate  # On Unix: source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

## Test-Driven Development

FatPy follows Test-Driven Development (TDD) principles:

1. Write tests that define expected behavior
2. Verify tests fail (red phase)
3. Implement code to make tests pass (green phase)
4. Refactor while maintaining test coverage
5. Repeat

For more details, see the [Testing Guide](https://vybornak2.github.io/FatPy/development/testing/).

## Coding Standards

This project uses:

- **Ruff** - Linting and formatting
- **MyPy** - Type checking
- **Pre-commit** - Automated quality checks

All code should be typed, documented, and follow the [Code Style Guide](https://vybornak2.github.io/FatPy/development/code_style/).

### Running Code Quality Checks

```bash
# Run linting
ruff check .

# Apply fixes automatically
ruff check --fix .

# Format code
ruff format .

# Run type checking
mypy .

# Run pre-commit on all files
pre-commit run --all-files
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/test_specific_module.py

# Test with coverage
pytest --cov=src/fatpy --cov-report=html
```

## Documentation

- Update documentation for API changes
- Add docstrings following Google style to all new code
- Include examples and mathematical formulas where helpful
- For detailed guidelines, see the [Documentation Guide](https://vybornak2.github.io/FatPy/development/documentation/)

### Building Documentation

```bash
# Build and serve documentation locally
mkdocs serve
```

## Pull Request Process

1. Ensure tests pass and code quality checks succeed
2. Update relevant documentation
3. Link any related issues
4. Make sure your code follows the project's style guidelines
5. Wait for review feedback

## Questions?

If you encounter issues or need assistance with development, you can:

- Create an issue in the [GitHub repository](https://github.com/vybornak2/fatpy/issues)
- Contact the maintainers at <jan.vyborny2@gmail.com>

Thank you for contributing to FatPy!
