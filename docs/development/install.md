---
title: Installation Guide
---

# :fontawesome-solid-download: Installation Guide

## :fontawesome-solid-list-check: Prerequisites

=== "General"
    - Python 3.12 or higher
    - Git
    - pip or uv package manager

=== "Windows"
    - Python 3.12 or higher (from [python.org](https://www.python.org/downloads/) or Microsoft Store)
    - Git for Windows (for development installation)
    - Optional: [uv](https://github.com/astral-sh/uv) package manager

=== "Linux"
    ```bash
    # Install Python 3.12+ (Ubuntu/Debian example)
    sudo apt update
    sudo apt install python3 python3-pip git

    # Install uv (optional)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "macOS"
    ```bash
    # Using Homebrew
    brew install python
    brew install git

    # Install uv (optional)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

??? tip
    Verify that the prerequisites are installed before setting up FatPy, by running following commands:
    ```bash
    uv --version
    uv python list
    git --version
    ```

## :material-cloud-download: Standard Installation

The easiest way to install FatPy is from PyPI:

=== "Using pip"

    ```bash
    pip install fatpy
    ```

=== "Using uv"

    Faster than standard pip installation.

    ```bash
    uv pip install fatpy
    ```
    ??? tip
        To install package and add dependency to your project .toml file use:

        ```bash
        uv add fatpy # Install from PyPI and add dependency 
        ```

This installs the latest stable release with all dependencies.

## :material-tools: Development Installation

For contributing or customizing FatPy:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/faberorg/fatpy.git
   cd fatpy
   ```

2. **Setup development environment:**

=== "Using uv (recommended)"
    ```bash
    # Create and activate virtual environment
    uv venv
    .venv\Scripts\activate  # On Unix: source .venv/bin/activate

    # Install dependencies
    uv sync

    # Install in development mode
    uv pip install -e .

    # Setup pre-commit hooks
    pre-commit install
    ```

    The uv package manager is significantly faster than pip and provides better dependency resolution.

=== "Using pip"
    ```bash
    # Create and activate virtual environment
    python -m venv venv
    venv\Scripts\activate  # On Unix: source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    pip install -r requirements-dev.txt

    # Install in development mode
    pip install -e .

    # Setup pre-commit hooks
    pre-commit install
    ```

    The standard pip approach works on all systems with Python installed.

## :material-check-circle: Verifying Installation

Run a simple test to verify the installation:

=== "Quick Test"
    ```bash
    # Import the library in Python
    python -c "import fatpy; print(fatpy.__version__)"
    ```

=== "Run Tests"
    ```bash
    # Run the test suite
    pytest -xvs
    ```
