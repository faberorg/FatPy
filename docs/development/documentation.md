---
title: Documentation
---

# Documentation Guide

Good documentation is crucial for the usability and maintainability of FatPy. This guide outlines our documentation standards and practices.

## Documentation Structure

FatPy's documentation is structured as follows:

1. **API Reference** - Detailed documentation of modules, classes, and functions
2. **Theory Reference** - Mathematical and physical background for implemented methods
3. **Development Guide** - Information for contributors
4. **Tutorials and Examples** - How to use the library

## Docstrings

All modules, classes, and functions must have docstrings following the Google style:

```python
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

## Building Documentation

We use MkDocs with the Material theme and mkdocstrings for API documentation:

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-autorefs

# Build documentation locally
mkdocs build

# Serve documentation locally with hot-reloading
mkdocs serve
```

Visit `http://127.0.0.1:8000` to view the documentation locally.

## Writing Documentation

### General Guidelines

- Use clear, concise language
- Include examples where possible
- Link to related documentation
- Use headers to organize content
- Include mathematical formulas using LaTeX when appropriate

### Mathematical Notation

Use LaTeX for mathematical formulas:

```markdown
$$ \sigma_{eq} = \sqrt{3J_2} = \sqrt{\frac{3}{2}s_{ij}s_{ij}} $$
```

### API Documentation

API documentation is automatically generated from docstrings using mkdocstrings. For this to work properly:

1. All public functions, classes, and modules must have docstrings
2. Type hints should be used for all function parameters and return values
3. Examples should be included in docstrings where appropriate
4. Mathematical formulas should use LaTeX syntax within docstrings

### Adding New Pages

1. Create a new Markdown file in the appropriate directory
2. Add the file to the navigation in `mkdocs.yml`
3. Use proper formatting and structure (headers, code blocks, etc.)

## Documentation Deployment

Documentation is automatically deployed using GitHub Actions when changes are pushed to the documentation branch. The workflow is defined in `.github/workflows/deploy_docs.yml`.

## Best Practices

- Update documentation when you change code
- Write documentation as you code, not after
- Test documentation examples to ensure they work
- Review documentation for clarity and correctness
- Consider the reader's perspective and knowledge level

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
