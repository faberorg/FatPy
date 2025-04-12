# CI/CD Process for FatPy

## Overview

FatPy uses GitHub Actions to automate testing, validation, documentation building, and deployment processes. This continuous integration and deployment (CI/CD) workflow ensures code quality and simplifies releases.

## CI Pipeline Components

### 1. Python CI

**Triggered by:**
- Pushes to `main` branch
- Pull requests to `main` branch

**Configuration file:** `.github/workflows/python-ci.yml`

**Steps:**

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

**Steps:**

1. **Checkout repository**: Retrieves the repository code
2. **Setup Python**: Configures Python environment
3. **Install dependencies**: Installs documentation tools
4. **Deploy**: Builds and deploys documentation to GitHub Pages

### 3. PyPI Publication

**Triggered by:**
- Release tags (v*.*.*)

**Configuration file:** `.github/workflows/publish_pypi.yml`

**Steps:**

1. **Checkout code**: Retrieves the repository code
2. **Setup Python**: Configures Python environment
3. **Install dependencies**: Installs build tools
4. **Build package**: Creates distribution packages
5. **Publish to PyPI**: Uploads built packages to PyPI

### 4. GitHub Release Creation

**Triggered by:**
- Release tags (v*.*.*)

**Configuration file:** `.github/workflows/github_release.yml`

**Steps:**

1. **Checkout**: Retrieves the repository code
2. **Create GitHub Release**: Creates a new release on GitHub

## Release Process

1. **Update version**: In `pyproject.toml`
2. **Update changelog**: Document changes
3. **Merge to main**: Ensure all changes are in the main branch
4. **Create tag**: `git tag -a v0.1.0 -m "Release v0.1.0"`
5. **Push tag**: `git push origin v0.1.0`

Pushing the tag automatically triggers:
- Package publishing to PyPI
- GitHub release creation

## Viewing Results

### CI/CD Status

1. Go to the GitHub repository
2. Click on the "Actions" tab
3. Select a workflow run to see detailed results

### Status Badges

Status badges are displayed in the README.md:

```markdown
![Python CI](https://github.com/vybornak2/fatpy/workflows/Python%20CI/badge.svg)
![Documentation](https://github.com/vybornak2/fatpy/workflows/Deploy%20Documentation/badge.svg)
```

## Troubleshooting CI Failures

### Common Issues and Solutions

#### Failed Tests
- **Issue**: Tests failing in CI but passing locally
- **Solutions**:
  - Check Python version differences
  - Check dependency versions
  - Review test logs for environment-specific issues

#### Type Errors
- **Issue**: Mypy reports type errors
- **Solutions**:
  - Run `mypy` locally: `mypy .`
  - Fix type annotations
  - Add appropriate type stubs if needed

#### Style Violations
- **Issue**: Ruff reports style issues
- **Solutions**:
  - Run `ruff check .` locally
  - Fix style issues: `ruff check --fix .`
  - Format code: `ruff format .`

#### Documentation Build Failures
- **Issue**: MkDocs build fails
- **Solutions**:
  - Run `mkdocs build --strict` locally
  - Check for broken links
  - Verify markdown syntax

## Adding New Workflows

To add a new workflow:

1. Create a YAML file in `.github/workflows/`
2. Define the workflow triggers and steps
3. Test the workflow using `workflow_dispatch` if possible

## Security Considerations

- **Secrets**: Sensitive information is stored as GitHub secrets
- **Token Access**: PyPI token has limited scope
- **Dependencies**: Regular updates to minimize vulnerabilities

## Best Practices

- **Test locally**: Run checks locally before pushing
- **Small changes**: Make smaller, focused changes
- **Review logs**: Check CI logs to understand failures
- **Update documentation**: Update CI documentation when changing workflows
