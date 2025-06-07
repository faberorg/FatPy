---
title: Testing
---


# Test Driven Development

This guide outlines the testing approach and best practices for the FatPy project.

## Testing Philosophy

FatPy follows the principles of Test-Driven Development (TDD):

1. **Write the test first** - Define what the code should do before implementing it
2. **See the test fail** - Run the test to confirm it fails without the implementation
3. **Write the minimal code** - Implement just enough code to make the test pass
4. **Run the test** - Verify the implementation meets the requirements
5. **Refactor** - Clean up the code while ensuring tests still pass

## Testing Framework

FatPy uses [pytest](https://docs.pytest.org/) for testing. The testing configuration can be found in the `pyproject.toml` file.

## Test Structure

Tests are organized in the `tests/` directory, mirroring the structure of the `src/fatpy` package:

```
tests/
├── core/
│   ├── stress_life/
│   │   ├── test_base_methods.py
│   │   └── ...
│   ├── strain_life/
│   └── energy_life/
├── data_parsing/
│   └── ...
├── utilities/
│   └── ...
└── conftest.py       # Shared fixtures and configuration
```

## Writing Tests

### Basic Test Structure

```python
# Test a function
def test_addition():
    # Arrange
    a = 2.0
    b = 3.0
    expected = 5.0

    # Act
    result = addition(a, b)

    # Assert
    assert result == expected


# Test a class
def test_some_class_multiply():
    # Arrange
    value = 5.0
    instance = SomeClass(value)
    factor = 2.0
    expected = 10.0

    # Act
    result = instance.multiply(factor)

    # Assert
    assert result == expected
```

### Test Naming

- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Test classes should be named `Test*`

### Fixtures

Use fixtures for setup and teardown:

```python
import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        "stress": [100.0, 200.0, 150.0],
        "cycles": [1000, 100, 500]
    }


def test_function_with_fixture(sample_data):
    # The fixture is automatically passed to the test
    result = process_data(sample_data["stress"], sample_data["cycles"])
    assert result > 0
```

### Parameterized Tests

Use parameterization to test multiple cases:

```python
import pytest


@pytest.mark.parametrize("input_value, expected_output", [
    (0.0, 0.0),
    (1.0, 1.0),
    (2.0, 4.0),
    (3.0, 9.0),
])
def test_square_function(input_value, expected_output):
    assert square(input_value) == expected_output
```

### Testing Exceptions

Test that functions raise appropriate exceptions:

```python
import pytest


def test_division_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10.0, 0.0)
```

## Testing Categories

### Unit Tests

- Test individual functions and methods
- Mock dependencies
- Should be fast and isolated

### Integration Tests

- Test interactions between components
- Use fewer mocks
- Verify that components work together correctly

### Numerical Tests

For mathematical functions, use appropriate numerical testing techniques:

```python
def test_numerical_function():
    result = calculate_value(3.14159)
    expected = 2.71828
    # Use pytest.approx for floating-point comparisons
    assert result == pytest.approx(expected, rel=1e-5)
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/test_specific_module.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src/fatpy

# Run specific test
pytest tests/core/test_module.py::test_specific_function
```

## Code Coverage

FatPy aims for high test coverage. Coverage reports can be generated with:

```bash
pytest --cov=src/fatpy --cov-report=html
```

Open `htmlcov/index.html` to view the coverage report.

## Best Practices

1. **Keep tests simple** - Each test should verify one specific behavior
2. **Use descriptive names** - Test names should describe what's being tested
3. **Avoid test interdependence** - Tests should not depend on each other
4. **Clean up after tests** - Use fixtures for setup and teardown
5. **Test edge cases** - Include tests for boundary conditions and error handling
6. **Keep tests fast** - Slow tests discourage frequent testing
7. **Use appropriate assertions** - Choose the right assertion for each test case
8. **Don't test implementation details** - Test behavior, not implementation

## Continuous Integration

Tests are automatically run on GitHub Actions when code is pushed or a pull request is created. See the [CI/CD guide](ci_cd.md) for more information.
