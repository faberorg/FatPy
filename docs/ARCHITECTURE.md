# FatPy Architecture

## Overview

FatPy follows a modular architecture for fatigue life evaluation of materials, with specialized modules for different analysis approaches.

## Directory Structure

```
src/
├── fatpy/
    ├── core/                  # Core analytical methods
        ├── stress_life/       # Stress-life analysis
        ├── strain_life/       # Strain-life analysis
        ├── energy_life/       # Energy-life analysis
    ├── data_parsing/          # Input/output handling
    ├── utilities/             # Helper functions and tools
```

## Core Modules

### Stress-Life Module

- Methods for analyzing fatigue life based on stress data
- Includes correction factors, decomposition methods, and damage parameter calculations

### Strain-Life Module

- Tools for evaluating fatigue life using strain data
- Includes models for elastic and plastic strain

### Energy-Life Module

- Methods for assessing fatigue life based on energy data
- Energy-based damage accumulation approaches

## Data Parsing

- Material data handling
- FE model integration
- Load case processing
- User input management

## Utilities

- Data transformation tools
- Pre-processing utilities
- Post-processing and visualization

For a visual representation of the architecture, see the [component diagram](ARCHITECTURE_GRAPH.md).
