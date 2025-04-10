# Installation Guide

## Prerequisites

- Python 3.13 or higher
- pip or uv package manager

## Installation Options

### Standard Installation

```bash
pip install fatpy
```

### Development Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vybornak2/fatpy.git
   cd fatpy
   ```

2. **Setup using uv (recommended):**
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

3. **Alternative setup using pip:**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   venv\Scripts\activate  # On Unix: source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

   # Install in development mode
   pip install -e .
   ```

## Verifying Installation

Run a simple test to verify the installation:

```bash
pytest -xvs
```

For more information, see the [Contributing Guide](../CONTRIBUTING.md) and [Usage Guide](USAGE.md).
