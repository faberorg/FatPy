# CI/CD Process for FatPy

## Overview

This project uses GitHub Actions to automatically run tests, type checks, and code quality checks whenever code is pushed or a pull request is created.

## CI Pipeline

The basic workflow includes:

1. **Trigger**: Runs on pushes to `main` branch and pull requests
2. **Setup**: Configures Python environment
3. **Quality Checks**:
   - Running tests with pytest
   - Type checking with mypy
   - Linting with ruff

## Viewing Results

1. Go to the GitHub repository
2. Click on the "Actions" tab
3. Select a workflow run to see detailed results

## Troubleshooting CI Failures

Common issues include:
- Failed tests: Check the test output in the workflow logs
- Type errors: Run `mypy` locally to identify issues
- Style violations: Run `ruff check .` locally to fix issues

For more details on testing locally, see the [Contributing Guide](../CONTRIBUTING.md).
