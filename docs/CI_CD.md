# CI/CD Process for FatPy

## Overview

This project uses GitHub Actions to automate the Continuous Integration (CI) and Continuous Deployment (CD) process. The CI/CD pipeline is configured to run tests, perform type checks, and lint the codebase every time a change is pushed to the `main` branch or a pull request is created.

## Workflow File

The CI/CD pipeline is defined in the `.github/workflows/python-ci.yml` file. Below is an overview of the steps included in the workflow:

1. **Trigger**: The workflow is triggered on pushes and pull requests to the `main` branch.
2. **Checkout Code**: The code is checked out from the repository.
3. **Set up Python**: A Python environment is set up using the specified Python version.
4. **Install Dependencies**: Project dependencies are installed using `pip`.
5. **Run Tests**: Tests are run using `pytest`.
6. **Run Type Checks**: Type checks are performed using `mypy`.
7. **Run Linter**: Linting is performed using `ruff`.

## Workflow Steps

### 1. Trigger

The workflow is triggered on the following events:
- Pushes to the `main` branch
- Pull requests targeting the `main` branch

### 2. Checkout Code

The code from the repository is checked out using the `actions/checkout@v2` action.

### 3. Set up Python

The Python environment is set up using the `actions/setup-python@v2` action. The Python version specified is `3.x`.

### 4. Install Dependencies

The dependencies are installed using `pip`. The command checks if a `requirements.txt` file exists and installs the dependencies listed in it.

### 5. Run Tests

The tests are run using `pytest`. This step ensures that the code changes do not introduce any new issues.

### 6. Run Type Checks

Type checks are performed using `mypy`. This step ensures that the code adheres to the specified type annotations.

### 7. Run Linter

Linting is performed using `ruff`. This step ensures that the code follows the project's coding standards and style guidelines.

## Viewing Workflow Results

To view the results of the workflow runs:

1. Navigate to the repository on GitHub.
2. Click on the "Actions" tab at the top of the repository page.
3. Click on the specific workflow run to see the details and logs for each step.

## Troubleshooting

If any of the steps in the workflow fail, you can review the logs for more information. Common issues might include missing dependencies, syntax errors, or misconfigurations in the workflow file. Make the necessary adjustments and push the changes to trigger the workflow again.

## Conclusion

By following this CI/CD process, we can ensure that our codebase remains stable and high-quality. If you have any questions or need further assistance, please reach out to the project maintainers.

---

**Note**: This documentation should be updated as the CI/CD process evolves. Ensure that any changes to the workflow file are reflected in this documentation.