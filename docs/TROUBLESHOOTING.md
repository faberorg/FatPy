# Troubleshooting Guide

## Development Issues

### Environment Setup Problems

**Symptoms**:
- Error when creating virtual environment
- Dependency installation fails

**Solution**:
- Ensure you have Python 3.13 or higher installed
- Try using uv for dependency management: `pip install uv`
- Check your internet connection when installing packages

### Code Quality Checks Failing

**Symptoms**:
- Pre-commit hooks fail
- CI tests fail

**Solution**:
- Run `ruff check --fix .` to automatically fix style issues
- Ensure all functions have type hints: `mypy .`
- Check test failures locally: `pytest -v`

### Import Errors

**Symptoms**:
- ModuleNotFoundError when running tests or application

**Solution**:
- Install the package in development mode: `uv pip install -e .`
- Check your Python path settings
- Verify the package structure matches [project structure](../README.md#project-structure)

## Getting Help

If you encounter issues not covered here:
- [Create an issue](https://github.com/vybornak2/fatpy/issues/new/choose) in the repository
- Contact the project maintainer via email at jan.vyborny2@gmail.com

For contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).
