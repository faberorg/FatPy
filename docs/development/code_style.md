# Code Style Guide

This guide outlines the coding standards and style guidelines for the FatPy project.

## General Principles

- **Readability** - Code should be easy to read and understand
- **Consistency** - Follow established patterns and conventions
- **Simplicity** - Prefer simple solutions over complex ones
- **Documentation** - Code should be well-documented

## Python Style Guidelines

FatPy follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some specific adaptations:

### Naming Conventions

- **Functions and variables**: `lowercase_with_underscores`
- **Classes**: `CamelCase`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Private attributes/methods**: `_leading_underscore`
- **"Magic" methods**: `__double_underscores__`

### Code Layout

- Line length: 120 characters maximum
- Indentation: 4 spaces (no tabs)
- Blank lines:
    - 2 between top-level functions and classes
    - 1 between methods in a class
    - Use blank lines to separate logical sections within functions

### Imports

- Standard library imports first, followed by third-party imports, followed by local application imports
- Each group should be separated by a blank line
- Within each group, imports should be alphabetized

```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party libraries
import numpy as np
import pandas as pd

# Local modules
from fatpy.core import analysis
from fatpy.utilities import helpers
```

## Type Annotations

FatPy uses type hints extensively. All functions should include type annotations:

```python
def calculate_stress(force: float, area: float) -> float:
    """Calculate stress from force and area.

    Args:
        force: The applied force in Newtons
        area: The cross-sectional area in square meters

    Returns:
        The stress in Pascals
    """
    return force / area
```

## Comments and Documentation

- Use docstrings for all modules, classes, and functions
- Use inline comments sparingly and only for complex or non-obvious code
- Keep comments up-to-date with code changes
- Follow Google's docstring style (see [Documentation Guide](documentation.md))

## Code Quality Tools

FatPy uses several tools to enforce code quality:

### Ruff

[Ruff](https://docs.astral.sh/ruff/) is used for linting and formatting:

```bash
# Run linting
ruff check .

# Apply fixes automatically
ruff check --fix .

# Format code
ruff format .
```

### MyPy

[MyPy](https://mypy.readthedocs.io/) is used for static type checking:

```bash
# Run type checking
mypy .
```

### Pre-commit

[Pre-commit](https://pre-commit.com/) runs checks automatically before each commit:

```bash
# Install the pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

## Best Practices

### General

- Keep functions and methods small and focused on a single task
- Limit function parameters to improve usability
- Use appropriate error handling and validation
- Write self-documenting code (clear variable names, logical structure)

### Performance

- Consider the computational complexity of your code
- Use vectorized operations with NumPy when working with numerical data
- Avoid premature optimization

### Testing

- Write tests for all new functionality
- Use descriptive test names that indicate what's being tested
- See the [Testing Guide](testing.md) for more details

## Examples

### Preferred Style

```python

import numpy as np
from numpy.typing import NDArray


def calculate_stress(forces: NDArray[np.float64], area: float) -> float:
    """Calculate stress from force and area.

    Args:
        forces: Numpy array of forces
        area: The cross-sectional area

    Returns:
        The stress value

    Raises:
        ValueError: If area is less than or equal to zero
    """

    if area <= 0:
        raise ValueError("Area must be greater than zero")

    total_force = np.sum(forces)

    return total_force / area
```

```python
class FatigueAnalyzer:
    """Class for performing fatigue analysis."""

    def __init__(self, material_name: str, safety_factor: float = 1.5) -> None:
        """Initialize the fatigue analyzer.

        Args:
            material_name: Name of the material to analyze
            safety_factor: Safety factor to apply in calculations
        """
        self.material_name = material_name
        self.safety_factor = safety_factor
        self._results: list[float] = []
```
