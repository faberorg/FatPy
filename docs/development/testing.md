---
title: Testing
---


# :fontawesome-solid-vial: Test Driven Development

This guide outlines the testing approach and best practices for the FatPy project.

## :fontawesome-solid-lightbulb: Testing Philosophy

FatPy follows the principles of Test-Driven Development (TDD):

1. **Write the test first**  
    Define what the code should do before implementing it
2. **See the test fail**  
    Run the test to confirm it fails without the implementation
3. **Write the minimal code**  
    Implement just enough code to make the test pass
4. **Run the test**  
    Verify the implementation meets the requirements
5. **Refactor**  
    Clean up the code while ensuring tests still pass
6. **Repeat**  
    Use the same principles for next feature

## :simple-pytest: Testing Framework

!!! info
    FatPy uses **[pytest](https://docs.pytest.org/)** for testing. The testing configuration can be found in the `pyproject.toml` file.

## :fontawesome-solid-folder-tree: Test Structure

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

## :material-text-box-edit: Writing Tests

### Basic Test Structure

```python
# Test a function
def test_addition():
    # Arrange: Set up inputs
    a = 2.0
    b = 3.0
    expected = 5.0

    # Act: Call the function
    result = addition(a, b)

    # Assert: Verify output
    assert result == expected

# Test a class
def test_some_class_multiply():
    # Arrange: Initialize class
    value = 5.0
    instance = SomeClass(value)
    factor = 2.0
    expected = 10.0

    # Act: Test method
    result = instance.multiply(factor)

    # Assert: Check result
    assert result == expected
```

!!! abstract "**Test Naming**"
    - Test files should be named `test_*.py`
    - Test functions should be named `test_*`
    - Test classes should be named `Test*`

---

### Fixtures

Use [fixtures](https://docs.pytest.org/en/stable/reference/fixtures.html) for setup and teardown:

```python
import pytest


@pytest.fixture
def sample_data(): # Fixture for sample data
    """Provide sample data for tests."""
    return {
        "stress": [100.0, 200.0, 150.0],
        "cycles": [1000, 100, 500]
    }

# The fixture is automatically passed to the test
def test_function_with_fixture(sample_data):
    # Act: Use fixture data
    result = process_data(sample_data["stress"], sample_data["cycles"])
    # Assert: Verify result
    assert result > 0
```

---

### Parameterized Tests

Use parameterization to test multiple cases:

```python
import pytest

# Define test cases
@pytest.mark.parametrize("input_value, expected_output", [
    (0.0, 0.0),
    (1.0, 1.0),
    (2.0, 4.0),
    (3.0, 9.0),
])
def test_square_function(input_value, expected_output):
    assert square(input_value) == expected_output
```

---

### Testing Exceptions

Use pytest to verify that functions raise appropriate exceptions:

```python
import pytest


def test_division_by_zero():
    # Assert: Expect ValueError
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10.0, 0.0)
```

## :fontawesome-solid-tags: Testing Categories

**FatPy uses multiple test types to ensure reliability:**

### Unit Tests

- Test individual functions and methods
- Mock dependencies
- Should be fast and isolated

---

### Integration Tests

- Test interactions between components
- Use fewer mocks
- Verify that components work together correctly

---

### Numerical Tests

For mathematical functions, use appropriate numerical testing techniques:

```python
def test_numerical_function():
    # Act: Compute value
    result = calculate_value(3.14159)
    expected = 2.71828
    # Assert: Compare floats
    assert result == pytest.approx(expected, rel=1e-5)
```

## :material-run-fast: Running Tests

**Run tests using pytest commands:**

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

## :fontawesome-solid-chart-pie: Code Coverage

FatPy aims for high test coverage. Coverage reports can be generated with:

```bash
pytest --cov=src/fatpy --cov-report=html
```

Open `htmlcov/index.html` to view the coverage report.

## :material-checkbox-marked-circle-outline: Best Practices

!!! note "Follow these practices to write effective tests for FatPy:"
    1. **Keep tests simple** - Each test should verify one specific behavior
    2. **Use descriptive names** - Test names should describe what's being tested
    3. **Avoid test interdependence** - Tests should not depend on each other
    4. **Clean up after tests** - Use fixtures for setup and teardown
    5. **Test edge cases** - Include tests for boundary conditions and error handling
    6. **Keep tests fast** - Slow tests discourage frequent testing
    7. **Use appropriate assertions** - Choose the right assertion for each test case
    8. **Don't test implementation details** - Test behavior, not implementation

## :material-sync-circle: Continuous Integration

Tests are automatically run on GitHub Actions when code is pushed or a pull request is created. See the FatPy’s [CI/CD](ci_cd.md) setup for more information.
