# Usage Guide

## Development Tools

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src/fatpy
```

### Code Quality Tools

```bash
# Run type checks
mypy .

# Run linter
ruff check

# Format code
ruff format
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit hooks manually
pre-commit run --all-files
```

## Getting Started with FatPy

This section will be expanded as the package develops. Currently, the package provides tools for fatigue life evaluation using different approaches:

- Stress-Life methods
- Strain-Life methods
- Energy-Life methods

Refer to the [project structure](../README.md#project-structure) for more details on the available modules.
