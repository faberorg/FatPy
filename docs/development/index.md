---
title: Development Guide
---

# :material-tools: Development Guide

This section provides comprehensive resources to help you contribute to FatPy, a Python package for fatigue life evaluation of materials, part of the FABER project.

[:material-search-web: Learn About FABER](../faber_cost.md)

!!! abstract "About"
    FatPy follows Test-Driven Development (TDD) principles, emphasizing code quality, thorough documentation, and rigorous testing. Whether you're fixing bugs or adding features, this guide will help you become a successful contributor.

## :fontawesome-solid-rocket: Getting Started

New to FatPy? Start with these resources to set up your environment and understand how to contribute effectively:

- :fontawesome-solid-download: **[Installation Guide](install.md)**  
  Set up your development environment for FatPy.
- :fontawesome-solid-handshake: **[Contributing Guide](contributing.md)**  
  Learn how to contribute code, documentation, or ideas to the project.
- :fontawesome-solid-book: **[Code of Conduct](code_of_conduct.md)**  
  Understand our community standards for respectful collaboration.

## :fontawesome-solid-code: Development Resources

Deepen your understanding of FatPy development with these resources:

- :fontawesome-solid-code: **[Code Style](code_style.md)**  
  Follow our coding standards for consistent, high-quality code.
- :fontawesome-solid-file-pen: **[Documentation](documentation.md)**  
  Learn best practices for writing and maintaining FatPy documentation.
- :fontawesome-solid-vial: **[Testing](testing.md)**  
  Understand how to write and run tests to ensure code reliability.
- :material-cogs: **[CI/CD Process](ci_cd.md)**  
  Explore our continuous integration and deployment workflow.

??? tip "Additional resources"
    - Join our [GitHub Discussions](https://github.com/vybornak2/FatPy/discussions) for community support.
    - Find contribution ideas in our [GitHub Issues](https://github.com/vybornak2/FatPy/issues).

## :fontawesome-solid-sitemap: Project Structure

FatPy is organized into several core modules:

- **Core** - Contains the fundamental analytical methods:
  - Stress-Life: Methods for stress-based fatigue analysis
  - Strain-Life: Methods for strain-based fatigue analysis
  - Energy-Life: Methods for energy-based fatigue analysis
- **Data Parsing** - Tools for handling input/output operations
- **Utilities** - Supporting functionality and helper methods

Get a detailed overview of FatPy’s modules and functions, visit **[API Reference](../api/index.md)**.

## :fontawesome-solid-code-branch: Development Workflow

Follow our Test-Driven Development (TDD) workflow:

1. **Write Tests First**  
   Define expected behavior before coding.
2. **Set Up Your Environment**  
   :fontawesome-solid-download: [Installation Guide](install.md)
3. **Create a Feature Branch**  

    ```bash
    git checkout -b feature-name
    ```

4. **Implement the Feature**  
    Follow the [Code Style Guide](code_style.md).
5. **Run and Refine Tests**  
    :fontawesome-solid-vial: [Testing Guide](testing.md)
6. **Document your Changes**  
    Use [Documentation](documentation.md) resources!
7. **Submit a pull request**  
    :fontawesome-solid-handshake: [Contributing Guide](contributing.md)

## :material-help-box-outline: Getting Help

Need assistance with FatPy development? Reach out through these channels:

- :simple-github:  **[GitHub Discussions](https://github.com/vybornak2/FatPy/discussions)**  
    Join for community support.
- :fontawesome-solid-bug: **[Report an Issue](https://github.com/vybornak2/fatpy/issues)**  
    Create an issue on our GitHub repository for bugs or questions.
- :fontawesome-solid-envelope: **[Contact our Team](../contact.md)**  
    Visit our contact page.

Thank you for contributing to FatPy!
