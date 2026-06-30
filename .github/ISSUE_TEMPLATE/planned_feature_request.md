---
name: Planned Implementation
about: Document a planned feature implementation for the project roadmap
title: ""
labels: ["WIP", "backlog"]
assignees: []

---
<!-- TEMPLATE GUIDELINE

MANDATORY blocks must be completed.
OPTIONAL blocks are provided inside HTML comments: `< !--`, `-- >`
    remove comment markers to enable

IMPORTANT:
Function docstrings are our documentation source of truth, so all relevant information should be included there.

Any important information captured in the issue should be reflected in the function signature, docstring, and/or test cases to ensure it is not lost during implementation.
-->

## ℹ️ General Information

<!-- MANDATORY: this section reflects current planning -->

**Component Name:** some_name

**Component Location:** src/FatPy/some_params.py

**Suggested Python Name:** calc_some_param <!-- should be verified against existing naming conventions, nomenclature table -->

**FABER WG Relation:** WG6.1

**Priority:** (1-10 scale)

**Technical Complexity:** (1-10 scale)

**Estimated Effort:** (1-10 scale)

**Dependencies:** <!-- List any dependencies on other components, libraries, or tools -->

**Related Issues:** <!-- Link to related issues, if any  using #Number_of_issue-->

---

<!-- MANDATORY -->
## 📋 Problem Description

Problem description goes here.

### Mathematical Formulation

<!-- Describe algorithms, equations, and mathematical relationships -->
$\text{Simple mathematical formulation here}$

$$
\text{Or more complex} \\
\text{mathematical formulation here}
$$

**Latex text:**
<!-- copy the formula inside the code block below for easier copy-pasting -->
```plaintext
$\text{Simple mathematical formulation here}$

$$
\text{Or more complex} \\
\text{mathematical formulation here}
$$
```

<!-- MANDATORY -->
## 🔧 Implementation Guideline


<!-- OPTIONAL
### Abstraction / Interface Design

Reference existing abstraction if any (base class, protocol, interface, abstract method).
If no abstraction exists, explicitly state: "No existing abstraction applies."

**Abstraction Reference:**
Example: src/fatpy/core/strain_life/base_methods.py::BaseMethod

-->

### Function Signature

<!--
Change function signature
- Use type hints for all parameters and return types
- Include default values for optional parameters
- Provide a docstring that describes the function's purpose, parameters, return value, and any
    exceptions it may raise
-->

```python
import numpy as np
from numpy.typing import ArrayLike, Optional, NDArray

def function_name(
    input1: ArrayLike,
    input2: Optional[ArrayLike] = None,
    param1: float = 0.5,
    param2: int = 10
) -> NDArray[np.float64]:
    """Brief description of the function.

    Some more detailed description of the function's purpose and behavior.

    Args:
        input1: Description of the first input.
        input2: Description of the second input (optional).
        param1: Description of the first parameter (default is 0.5).
        param2: Description of the second parameter (default is 10).

    Returns:
        Description of the output.
    
    Raises:
        ValueError: Description of when this error is raised.
        TypeError: Description of when this error is raised.    
    """
    pass  # Implementation goes here
```

<!-- OPTIONAL
### Inputs

| Parameter | Symbol | Type | Description | Units | Constraints |
|-----------|--------|------|-------------|-------|-------------|
|           |        |      |             |       |             |

-->


<!-- OPTIONAL
### Outputs

| Parameter | Symbol | Type | Description | Units | Range |
|-----------|--------|------|-------------|-------|-------|
|           |        |      |             |       |       |

-->

<!-- OPTIONAL
### Expected Behavior

- expected behavior
- input assumptions
- computation path
- boundary behavior
- additional requirements

<!--

<!-- OPTIONAL
### Error Handling

Document validation and failure behavior if applicable:
- Input validation rules
- Raised exceptions and when they occur
- Message style for user-facing errors
- Behavior for non-fatal issues (warning vs error)

--> 

<!-- MANDATORY -->
## ✅ Validation & Testing

#### Test Cases

<!-- Provide specific test cases with expected results -->

| Test Case | Inputs | Expected Outputs | Notes |
|-----------|--------|------------------|-------|
| Example 1 |        |                  |       |
| Example 2 |        |                  |       |

### Acceptance Criteria
<!-- Define clear acceptance criteria for the implementation -->
- [ ] Implementation meets all specified requirements
- [ ] Code is well-documented and follows style guidelines
- [ ] All test cases pass successfully
<!-- If additional criteria are needed, list them here -->

<!-- MANDATORY -->
## 📚 References & Resources

<!-- MANDATORY: literature, papers, existing implementations, theory -->
<!-- Ideally provide links to online resources -->
- [Reference 1](https://example.com)

