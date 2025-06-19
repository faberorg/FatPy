---
title: Documentation
---

# :fontawesome-solid-file-pen: Documentation Guide

Good documentation ensures FatPy is usable and maintainable for researchers, engineers, and contributors. This guide outlines our standards to support FatPy’s role in the FABER project.

## :fontawesome-solid-list: Documentation Structure

FatPy’s documentation is organized into the following sections to support diverse user needs:

=== "**:fontawesome-solid-book-open: API Reference**"
    <div style="text-align: center;">
    Detailed documentation for modules, classes, and functions.  
    <br>&nbsp;
    [:fontawesome-solid-book-open: API Reference](../api/index.md){ .md-button }
    </div>

=== "**:fontawesome-solid-square-root-variable: Theory Reference**"
    <div style="text-align: center;">
    Mathematical and physical background for implemented fatigue analysis methods.  
    <br>&nbsp;
    [:fontawesome-solid-square-root-variable: Theory Documentation](../theory/index.md){ .md-button }  
    </div>

=== "**:material-tools: Development Guide**"  
    <div style="text-align: center;">
    Resources for contributors, including setup and contribution guidelines.  
    <br>&nbsp;
    [:material-tools: Development Guide](index.md){ .md-button }  
    </div>

=== "**:fontawesome-solid-graduation-cap: Tutorials and Examples**"  
    <div style="text-align: center;">
    Practical guides and examples for using FatPy.  
    <br>&nbsp;
    [:fontawesome-solid-graduation-cap: Tutorials](#){ .md-button }  
    </div>

## :material-code-equal: Docstrings

!!! Note
    All modules, classes, and functions must include Google-style docstrings with:

      - A short description
      - Args, Returns, Raises sections
      - Optional LaTeX formulas and code examples
      - Detailed explanation (if needed)

```py title="example_module.py"
def example_function(param1: int, param2: str) -> bool:
    """Short description of the function.

    More detailed explanation if needed. This can span
    multiple lines and include more information.

    Mathematical formulas can be included using LaTeX syntax:

    $$ y = f(x) $$

    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter

    Returns:
        Description of the return value

    Raises:
        ValueError: When an invalid value is provided

    Example:
        ```python
        result = example_function(42, "test")
        print(result)  # Output: True
        ```
    """
    # Function implementation
    return True
```

## :fontawesome-solid-hammer: Building Documentation

We use MkDocs with the Material theme and mkdocstrings for API documentation, see [references :octicons-arrow-down-24:](#documentation-resources)

You can build and preview FatPy’s documentation locally using MkDocs commands:

```bash
# Activate project environment
.\{environment_name}\Scripts\activate

# Install documentation dependencies
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-autorefs

# Build documentation locally
mkdocs build

# Serve documentation locally with hot-reloading
mkdocs serve
```

The `#!bash mkdocs serve` command will provide you with link `http://127.0.0.1:8000` to view the documentation locally.

## :fontawesome-solid-pen-nib: Writing Documentation

Follow these guidelines to create clear and effective documentation for FatPy.

### **General Guidelines**

- **Clear Language**  
  Use concise, straightforward wording.
- **Examples**  
  Include practical code or use-case examples, where possible.
- **Related Links**  
  Reference related documentation.
- **Headers**  
  Organize content with clear section headings.
- **Mathematical Formulas**  
  Use LaTeX for mathematical notation where appropriate.

---

### **FatPy Naming Conventions**

Consistent naming is crucial for code readability and maintainability. Follow these conventions alongside the standard Python's PEP 8 style when contributing to FatPy.

=== ":material-function: Method & Function Naming"

    - **Use `snake_case`**: All function and method names should be lowercase with words separated by underscores.
        ```python
        def calc_stress_von_mises():
            pass
        ```
    - **Calculation functions**: utilize `calc_` prefix in the function name signature.

=== ":material-variable: Parameter & Variable Naming"

    - **Use `snake_case`**: All parameter and variable names should be lowercase with words separated by underscores.
        ```python
        def example_function(stress_amp, cycle_count):
            mean_stress = 0  # Example variable
            # ...
        ```
    - **Clarity over Brevity**: Choose names that are easy to understand.
        - *Good*: `eq_stress_amp` (see common abbreviations and shortcuts)
        - *Avoid*: `s`, `val`, `x1` (if context is not immediately obvious)
    - **Units (Optional but Recommended in Docstrings)**: While not part of the variable name itself, always specify units in docstrings for physical quantities.
        ```python
        def calc_force(mass: float, acceleration: float) -> float:
            """Calculates force.

            $$ F = m \cdot a $$

            Args:
                mass: Mass of the object [kg]
                acceleration: Acceleration [$$ ms^2 $$]

            Returns:
                Force [N]
            """
            return mass * acceleration
        ```

=== ":material-format-letter-case: Common Abbreviations & Shortcuts"

    Use abbreviations sparingly and only when they are widely understood within the fatigue analysis domain or clearly defined within the project.

    - **`eq`**: equivalent (e.g., `eq_stress` for equivalent stress)
    - **`amp`**: amplitude (e.g., `stress_amp` for stress amplitude)
    - **`mean`**: mean (e.g., `mean_stress`)
    - **`max`**: maximum (e.g., `max_principal_stress`)
    - **`min`**: minimum (e.g., `min_principal_stress`)
    - **`fat`**: fatigue (e.g., `fat_limit` for fatigue limit)
    - **`frac`**: fracture (e.g., `frac_toughness` for fracture toughness)

    Common names and material parameters:  
    - `first_principal_stress`, `second_principal_stress` and `third_principal_stress`  
    - `elastic_modulus`, `shear_modulus` and `poisson_ratio`  
    - `ultimate_tensile_strength` and `yield_strength`  

    !!! warning "Clarity First"
        If an abbreviation makes the code harder to understand for someone new to the specific module or fatigue in general, prefer the full name. Document any project-specific abbreviations clearly.

Learn more about the naming conventions form our discussion page:

[:simple-github::fontawesome-solid-comment-dots: Naming Conventions for FatPy Functions & Parameters #18](https://github.com/Vybornak2/FatPy/discussions/18){ .md-button }

---

!!! abstract "**Best Practices**"

    - Update documentation when you change code.
    - Write documentation as you code, not after.
    - Test documentation examples to ensure they work.
    - Review documentation for clarity and correctness.
    - Consider the reader's perspective and knowledge level.

## :fontawesome-solid-square-root-variable: Mathematical Notation

Use LaTeX for mathematical formulas. This allows for clear rendering of equations. You can include math inline with text or as a separate display block.

=== "Inline Math"

    Inline blocks must be enclosed in `#!latex $...$`, and can be used within a sentence.

    **How to write it:**
    ``` latex title="Inline syntax"
    Hooke's Law can be expressed as $\sigma = E \cdot \epsilon$.
    ```

    **Rendered output:**  
    Hooke's Law can be expressed as $\sigma = E \cdot \epsilon$.

=== "Block Syntax"
    This is typically used for important standalone equations. Blocks must be enclosed in `#!latex $$...$$`, and separated by empty lines form other text.

    **How to write it:**
    ``` latex title="Block syntax"
    % This line is empty
    $$ 
    \sigma_{eq} = \sqrt{3J_2} = \sqrt{\frac{3}{2}s_{ij}s_{ij}}
    $$
    % This line is empty
    ```

    **Rendered output:**

    $$
    \sigma_{eq} = \sqrt{3J_2} = \sqrt{\frac{3}{2}s_{ij}s_{ij}}
    $$

=== "Display Complex Expression"
    For more complex expressions, like summations, matrices, or a sequence of aligned equations.  

    This example shows Miner's rule for damage accumulation:

    **How to write it:**
    ``` latex title="Sums or Integrals"
    $$
    \begin{gather}
    D = \sum_{i=1}^{k} \frac{n_i}{N_i} \\[2mm]
    \text{or} \\[2mm]
    D = \int_{0}^{N} \frac{1}{N_f(\sigma(t))} \, dt
    \end{gather}
    $$
    ```

    **Rendered output:**

    $$
    \begin{gather}
    D = \sum_{i=1}^{k} \frac{n_i}{N_i} \\[2mm]
    \text{or} \\[2mm]
    D = \int_{0}^{N} \frac{1}{N_f(\sigma(t))} \, dt
    \end{gather}
    $$

    !!! tip "Multiple lines and spacing"
        Use the `gather` environment to center equations, placing each on its own line. Separate lines with `\\`, and use `[2mm]` to adjust vertical spacing between them.

    ---

    **To achieve precise alignment** within equations, use aligned environment and the `&` character to mark alignment points:


    ``` latex title="Aligned syntax"
    $$
    \begin{aligned}
    (a+b)^2 &= (a+b)(a+b) \\
            &= a^2 + 2ab + b^2
    \end{aligned}
    $$
    ```

    **Rendered output:**

    $$
    \begin{aligned}
    (a+b)^2 &= (a+b)(a+b) \\
            &= a^2 + 2ab + b^2
    \end{aligned}
    $$

## :fontawesome-solid-book-open: API Documentation

API documentation is automatically generated from docstrings using mkdocstrings. For this to work properly:

1. All public functions, classes, and modules must have Google-style docstrings.
2. Type hints should be used for all function parameters and return values.
3. Examples should be included in docstrings where appropriate.
4. Mathematical formulas should use LaTeX syntax within docstrings, e.g.:  
    `$$ \sigma = \sqrt{x^2} $$`

View the generated API documentation:

**[:fontawesome-solid-book-open: API Reference](../api/index.md)**{.md-button}

## :material-file-plus: Adding New Pages

To add new documentation pages, follow these steps:

- **Create File**  
  Add a new Markdown file in the appropriate directory (e.g., `docs/`).
- **Update Navigation**  
  Include the file in `mkdocs.yml` under the `nav` section.
- **Format Content**  
  Use headers, code blocks, and consistent Markdown formatting.

??? note "Documentation Deployment"

    Documentation is automatically deployed via GitHub Actions when changes are pushed to the documentation branch. The workflow is defined in `.github/workflows/deploy_docs.yml`.

## Documentation Resources

<div class="grid cards" markdown>

- :fontawesome-brands-markdown:{ .lg .middle } **Markdown**

    ---

    Markdown, the simple and easy-to-use markup language you can use to format virtually any document.

    [:octicons-arrow-right-24: Cheat Sheet](https://www.markdownguide.org/cheat-sheet/) &emsp;&emsp;&emsp; [:octicons-arrow-right-24: Basic Syntax](https://www.markdownguide.org/basic-syntax/)

- :material-file-document-multiple:{ .lg .middle } **Markdown DocsString**

    ---

    In code documentation written in Markdown and automatically extracted into documentation.

    [:octicons-arrow-right-24: mkdocstring](https://mkdocstrings.github.io)

- :material-script-text-outline:{ .lg .middle } **MkDocs**

    ---

    MkDocs is a fast and simple static site generator that's geared towards building project documentation.

    [:octicons-arrow-right-24: MkDocs](https://www.mkdocs.org)

- :simple-materialformkdocs:{ .lg .middle } **Material for MkDocs**

    ---

    MkDocs Extension with various functionality including search, annotations, buttons and more!

    [:octicons-arrow-right-24: Materials for MkDocs](https://squidfunk.github.io/mkdocs-material/)

</div>
