---
title: CI/CD
---


# :material-cogs: CI/CD Process for FatPy

## :material-information-slab-circle: Overview

FatPy uses GitHub Actions to automate testing, validation, documentation building, and deployment processes. This continuous integration and deployment (CI/CD) workflow ensures code quality and simplifies releases.

## :material-pipe: CI Pipeline Components

### 1. Python CI

**Triggered by:**

- Pushes to `main` branch
- Pull requests to `main` branch

**Configuration file:** `.github/workflows/python-ci.yml`

??? note "**Steps:**"

    1. **Checkout code**: Retrieves the repository code
    2. **Setup Python**: Configures Python 3.12 environment
    3. **Install dependencies**: Installs required packages
    4. **Run type checks**: Validates typing with mypy
    5. **Run linter**: Checks code quality with ruff
    6. **Run tests**: Executes pytest test suite
    7. **Test documentation build**: Ensures docs build successfully

### 2. Documentation Deployment

**Triggered by:**

- Pushes to `documentation` branch
- Manual trigger via workflow_dispatch

**Configuration file:** `.github/workflows/deploy_docs.yml`

??? note "**Steps:**"

    1. **Checkout repository**: Retrieves the repository code
    2. **Setup Python**: Configures Python environment
    3. **Install dependencies**: Installs documentation tools
    4. **Deploy**: Builds and deploys documentation to GitHub Pages

### 3. PyPI Publication

**Triggered by:**

- Release tags (v*.*.*)

**Configuration file:** `.github/workflows/publish_pypi.yml`

??? note "**Steps:**"

    1. **Checkout code**: Retrieves the repository code
    2. **Setup Python**: Configures Python environment
    3. **Install dependencies**: Installs build tools
    4. **Build package**: Creates distribution packages
    5. **Publish to PyPI**: Uploads built packages to PyPI

### 4. GitHub Release Creation

**Triggered by:**

- Release tags (v*.*.*)

**Configuration file:** `.github/workflows/github_release.yml`

??? note "**Steps:**"

    1. **Checkout**: Retrieves the repository code
    2. **Create GitHub Release**: Creates a new release on GitHub

## :fontawesome-solid-rocket: Release Process

Follow these steps to release a new FatPy version:

1. **Update version**: In `pyproject.toml`
2. **Update changelog**: Document changes
3. **Merge to main**: Ensure all changes are in the main branch
4. **Create tag**: `git tag -a v0.1.0 -m "Release v0.1.0"`
5. **Push tag**: `git push origin v0.1.0`

!!! note
    Pushing the tag automatically triggers:

      - Package publishing to PyPI
      - GitHub release creation

## :fontawesome-solid-eye: Viewing Results

Monitor CI/CD outcomes to ensure successful workflows:

### CI/CD Status

1. Go to the GitHub repository
2. Click on the "Actions" tab
3. Select a workflow run to see detailed results

### Status Badges

Status badges are displayed in the README.md:

```markdown
![Python CI](https://github.com/faberorg/fatpy/workflows/Python%20CI/badge.svg)
![Documentation](https://github.com/faberorg/fatpy/workflows/Deploy%20Documentation/badge.svg)
```

## :octicons-tools-24: Troubleshooting CI Failures

### Common Issues and Solutions

=== "**Failed Tests**"
    - **Issue**: Tests failing in CI but passing locally
    - **Solutions**:
      - Check Python version differences
      - Check dependency versions
      - Review test logs for environment-specific issues
  
=== "**Type Errors**"

    - **Issue**: Mypy reports type errors
    - **Solutions**:
      - Run `mypy` locally: `mypy .`
      - Fix type annotations
      - Add appropriate type stubs if needed

=== "**Style Violations**"

    - **Issue**: Ruff reports style issues
    - **Solutions**:
      - Run `ruff check .` locally
      - Fix style issues: `ruff check --fix .`
      - Format code: `ruff format .`

=== "**Documentation Build Failures**"

    - **Issue**: MkDocs build fails
    - **Solutions**:
      - Run `mkdocs build --strict` locally
      - Check for broken links
      - Verify markdown syntax

## :material-tab-plus: Adding New Workflows

To add a new CI/CD workflows to enhance FatPy’s automation:

1. Create a YAML file in `.github/workflows/`
2. Define the workflow triggers and steps
3. Test the workflow using `workflow_dispatch` if possible

## :fontawesome-solid-lock: Security Considerations

!!! warning
    Ensure secure CI/CD practices for FatPy:

    - **Secrets**: Sensitive information is stored as GitHub secrets
    - **Token Access**: PyPI token has limited scope
    - **Dependencies**: Regular updates to minimize vulnerabilities

## :material-checkbox-marked-circle-outline: Best Practices

- **Test locally**: Run checks locally before pushing
- **Small changes**: Make smaller, focused changes
- **Review logs**: Check CI logs to understand failures
- **Update documentation**: Update CI documentation when changing workflows
