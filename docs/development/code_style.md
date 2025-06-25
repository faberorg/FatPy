---
title: Code Style
---


# :fontawesome-solid-code: Code Style Guide

This guide defines coding standards and style to ensure readable and consistent code for FatPy project.

## :fontawesome-solid-lightbulb: General Principles

- **Readability** - Code should be easy to read and to understand
- **Consistency** - Follow established patterns and conventions
- **Simplicity** - Prefer simple solutions over complex ones
- **Documentation** - Code should be well-documented

## :simple-python: Python Style Guidelines

FatPy follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some specific adaptations:

=== "**Naming Conventions**"

    - **Functions and variables**: `lowercase_with_underscores`
    - **Classes**: `CamelCase`
    - **Constants**: `UPPERCASE_WITH_UNDERSCORES`
    - **Private attributes/methods**: `_leading_underscore`
    - **"Magic" methods**: `__double_underscores__`

=== "**Code Layout**"

    - Line length: 120 characters maximum
    - Indentation: 4 spaces (no tabs)
    - Blank lines:
      - 2 between top-level functions and classes
      - 1 between methods in a class
      - Use blank lines to separate logical sections within functions

=== "**Imports**"

    - Organize imports as follows:
      - Standard library imports
      - Third-party imports
      - Local application imports
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

## :material-head-dots-horizontal: Type Annotations

!!! note
    FatPy uses type hints extensively. All functions should include type annotations to enhance code clarity

**Example:**

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

See more [examples :octicons-arrow-down-24:](#examples)

## :fontawesome-solid-comment-dots: Comments and Documentation

- **Docstrings** - Use Google-style for all modules, classes, and functions.
- **Inline comments** - Use sparingly and only for complex or non-obvious code.
- **Keep Updated** - Sync comments with code changes.
- **Style Guide** -  **[:fontawesome-solid-file-pen: Follow Documentation Guide](documentation.md)**

## :octicons-tools-16: Code Quality Tools

FatPy uses several tools to enforce code quality:

**[:simple-ruff: Ruff](https://docs.astral.sh/ruff/)**

Handles linting and formatting.

```bash
# Run linting
ruff check .

# Apply fixes automatically
ruff check --fix .

# Format code
ruff format .
```

---

**[:simple-python: MyPy](https://mypy.readthedocs.io/)**

Enforces static type checking.

```bash
# Run type checking
mypy .
```

---

**[:simple-precommit: Pre-commit](https://pre-commit.com/)**

Automates and run code checks before commits.

```bash
# Install the pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

## :material-checkbox-marked-circle-outline: Best Practices

### General

- Keep functions and methods small and focused on a single task
- Limit function parameters to improve usability
- Use appropriate error handling and validation
- Write self-documenting code (clear variable names, logical structure)

---

### Performance

- Consider the computational complexity of your code
- Use vectorized operations with NumPy when working with numerical data
- Avoid premature optimization

---

### Testing

- Write tests for all new functionality
- Use descriptive test names that indicate what's being tested
- For more details, see: [:fontawesome-solid-vial: Testing Guide](testing.md)

## :fontawesome-solid-graduation-cap: Examples

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
