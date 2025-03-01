# FatPy

FatPy is a Python package for fatigue life evaluation of materials.

## Features

- Feature 1

## Installation

### Prerequisites
- Python 3.11 or higher
- UV package manager

### Standard Installation
```bash
uv venv
uv pip install .
```

### Development Installation
```bash
# Create and activate virtual environment
uv venv
.venv\Scripts\activate

# Install in development mode with dev dependencies
uv pip install -e .

# Setup pre-commit hooks
pre-commit install
```

## Project Structure
```
src/
├── fatpy/
    ├── __init__.py
    ├── module1/
    ├── module2/
tests/
├── __init__.py
├── conftest.py
├── module1_tests/
├── module2_tests/
```

## Development

### Code Quality Tools
- Ruff for linting and formatting
- MyPy for type checking
- Pre-commit hooks for automated checks

### Running Tests
```bash
pytest
```

### Code Coverage
```bash
pytest --cov=src/fatpy
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Your Name - jan.vyborny2@gmail.com
Project Link: [FatPy](https://github.com/vybornak2/fatpy)
