# :handshake: Contributing to FatPy

Thank you for considering contributing to FatPy, a Python package for fatigue life evaluation of materials, part of the FABER project! Your contributions help improve open source tools for fatigue analysis. Start with these steps to make an impact.

**[:mag: Learn About FABER](https://vybornak2.github.io/FatPy/faber_cost/)**

## :rocket: Quick Start

Get started contributing to FatPy in a few simple steps:

1. **Fork the Repository**  
Create a personal copy on GitHub.
2. **Clone Your Fork**  

```bash
git clone https://github.com/your-username/FatPy.git  # Clone your fork
```

3. **Create a Branch**  

```bash
git checkout -b my-feature-branch  # Create a new branch for your feature
```

4. **Write Tests for Your Feature**  
Define expected behavior first.
5. **Implement the Feature**  
Write code to pass your tests.
6. **Submit a Pull Request**  
Share your changes for review.

## :hammer_and_wrench: Development Setup

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

For detailed configuration of your development environment, see the **[Installation Guide :arrow_right:](https://vybornak2.github.io/FatPy/development/install/#installation-guide)**

## :test_tube: Test-Driven Development

### FatPy follows Test-Driven Development (TDD) principles

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

For more details, see the **[Testing Guide :arrow_right:](https://vybornak2.github.io/FatPy/development/testing/)**

### Run tests regularly

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/test_specific_module.py

# Test with coverage
pytest --cov=src/fatpy --cov-report=html
```

## :computer: Coding Standards

FatPy aims for high code quality utilizing these tools:

- **Ruff**  - Linting and formatting for consistent code style.
- **MyPy**  - Static type checking for reliability.
- **Pre-commit** - Automated checks before commits.

Follow our coding standards for contributions, see **[Code Style Guide :arrow_right:](https://vybornak2.github.io/FatPy/development/code_style/)**  

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

## :memo: Documentation

Keep FatPy’s documentation clear and up-to-date with these guidelines:

- **API Changes**  
Update documentation for any API modifications.
- **Docstrings**  
Add docstrings following Google style to all new code.
- **Examples**  
Include examples and mathematical formulas where helpful to aid users.

Learn best practices and guidelines for documentation, see **[Documentation Guide :arrow_right:](https://vybornak2.github.io/FatPy/development/documentation/)**

### Building Documentation

```bash
# Activate project environment
.\{environment_name}\Scripts\activate

# Build and serve documentation locally
mkdocs serve
```

## :arrows_clockwise: Pull Request Process

Submit a high-quality pull request with these steps:

1. **Run Tests and Checks**  
   Ensure tests pass and code quality checks succeed.
2. **Update Documentation**  
   Reflect changes in relevant docs.
3. **Link Issues**  
   Reference related GitHub issues.
4. **Follow guidelines**  
   Make sure your code follows the project's style guidelines.
5. **Await Review**  
   Respond to feedback from maintainers.

## :book: Code of Conduct

All contributors must follow our standards.

Understand our expectations for collaboration, see **[Code of Conduct :arrow_right:](https://github.com/Vybornak2/FatPy/blob/main/CODE_OF_CONDUCT.md)**

## :bulb: Ideas and Questions

Join discussion, create an issue or reach out to maintainers:

- :speech_balloon:  **[GitHub Discussions :arrow_right:](https://github.com/vybornak2/FatPy/discussions)**  
    Join for community support.
- :beetle: **[Report an Issue :arrow_right:](https://github.com/vybornak2/fatpy/issues)**  
    Create an issue on our GitHub repository for bugs or questions.
- :envelope: **[Contact our Team :arrow_right:](https://vybornak2.github.io/FatPy/contact/)**  
    Visit our contact page.

Thank you for contributing to FatPy!
