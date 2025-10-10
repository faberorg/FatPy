---
title: Home
---

# FatPy

Welcome to FatPy, an open-source Python package for fatigue life evaluation of materials.

## :fontawesome-solid-circle-info: Introduction

FatPy is developed to assist researchers, engineers, and industry
professionals in accurately assessing material fatigue life. Its modular architecture is designed
to accommodate a variety of fatigue analysis methods, making it a versatile tool in the field of
material science.

### Key Features

#### Core Modules

- **Stress-Life**  
    Stress-based fatigue assessment methods for estimating fatigue damage with corrections for various effects (stress concentration, size, surface quality, etc.)
- **Strain-Life**  
    Fatigue analysis using strain amplitude and cycles to failure relationships (ε-N approaches) such as Coffin-Manson and Basquin laws, suited for low-cycle and transitional fatigue regimes
- **Energy-Life**  
    Fatigue analysis methods based on the relationship between strain energy density and number of cycles to failure
- **Damage Cumulation**  
    Various fatigue damage accumulation rules for variable amplitude loading, including linear models (Palmgren-Miner) and advanced non-linear approaches accounting for load sequence effects
- **Decompositions**  
    Methods for breaking down complex load signals into cycles, containing both uniaxial and multiaxial procedures
- **Plane-Based Methods**  
    Methods for processing stress tensor paths on material planes, providing infrastructure for critical-plane and integral prediction approaches

#### Supporting Modules

- **Material Laws**  
    Functions dealing with various material constitutive laws and behavior models
- **Structural Mechanics**  
    Stress analysis and transformation utilities
- **Utils**  
    General utilities for mesh handling, signal processing, and data manipulation

## :fontawesome-solid-download: Installation

=== "Using pip"

    Install from PyPI via pip

    ```bash
    pip install fatpy
    ```

=== "Using uv"

    Faster installation from PyPI

    ```bash
    uv pip install fatpy
    ```
    ??? tip
        To install package and add dependency to your project .toml file use:

        ```bash
        uv add fatpy # Install from PyPI and add dependency 
        ```

For detailed installation options, see the [Installation Guide :octicons-arrow-right-24:](development/install.md)

## The FABER Project

!!! abstract "Background"
    FatPy is a key initiative of Working Group 6 (WG6) within the FABER (Fatigue Benchmark Repository) project,
    operating under COST Action CA23109. The FABER project aims to create a comprehensive database of
    experimental fatigue data to enhance the accuracy of fatigue life predictions.

Working Group 6 is specifically focused on developing open-source fatigue software, with FatPy being
a flagship product of this endeavor.

### FABER Resources

<div class="grid cards" markdown>

- [:material-search-web: Official Website](https://faber-cost.eu/)
- [:fontawesome-solid-newspaper: FABER Newsletter](https://faber-cost.eu/media-newsletter/)
- [:fontawesome-brands-linkedin: LinkedIn Organization Page](https://www.linkedin.com/company/faber-cost/about/)
- [:fontawesome-brands-linkedin: LinkedIn Group Page](https://www.linkedin.com/groups/13170259/)

</div>

<div class="grid cards" style="text-align: center;" markdown>

- **[More about FABER :octicons-arrow-right-24:](faber_cost.md)**

</div>

## :fontawesome-solid-handshake: Join Our Efforts

We warmly invite you to contribute to the development and enhancement of FatPy. Whether you're
interested in coding, testing, documentation, or providing feedback, your participation is invaluable.

### Getting Involved

- **Explore the Repository**:  
Visit our [:fontawesome-brands-github: GitHub repository](https://github.com/faberorg/fatpy) to access the source code.

- **Contribute**:  
Follow our [:fontawesome-solid-handshake: Contribution Guidelines](development/contributing.md) to help improve FatPy.

- **Collaborate**:  
Join our [:fontawesome-solid-comment-dots: Discussions](https://github.com/faberorg/fatpy/discussions) or [:fontawesome-solid-bug: Report issues](https://github.com/faberorg/fatpy/issues).

By participating, you'll be part of a collaborative effort to advance the field of material fatigue analysis.
Stay tuned to the FatPy documentation for regular updates, tutorials, and community news as we work together
to push the boundaries of fatigue research.

## :material-scale-balance: License

This project is licensed under the [MIT License](license.md).

## :fontawesome-solid-envelope: Contact

Feel free to share your ideas, use these channels to contact us:

<div class="grid cards" markdown>

- [:fontawesome-solid-envelope: Contact page](contact.md)
- [:fontawesome-brands-github: Project GitHub repository](https://github.com/faberorg/fatpy)

</div>

## :material-account-heart: Special Thanks

[:fontawesome-brands-github: Gooby](https://github.com/jakubda1) - for his help with the documentation and the project in general.
