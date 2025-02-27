# Installation Guide

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have a Windows/Linux/Mac machine.
- You have Python 3.x installed.

## Installation Steps

### Option 1: Using pip

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vybornak2/FatPy.git
   cd FatPy
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Option 2: Using uv

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vybornak2/FatPy.git
   cd FatPy
   ```

2. **Install the uv package manager:**
   ```bash
   pip install uv
   ```

3. **Create a virtual environment using uv:**
   ```bash
   uv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Synchronize the dependencies using uv:**
   ```bash
   uv sync
   ```

## Running the Application

```bash
python main.py  # Adjust this command to your project's entry point
```

## Additional Setup

- **Database setup**: (if applicable)
- **Environment variables**: (if applicable)