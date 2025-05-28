"""This module contains an example function with a detailed docstring."""


def example_function_with_docstring(a: int, b: int) -> int:
    r"""This docstring highlights Mermaid diagrams, math expressions, and code examples.

    This function takes two integers and returns their sum.
    It demonstrates various mkdocs-material features:

    !!! info "Information"
        This is an informational admonition block.

    !!! warning "Important Note"
        Make sure both parameters are integers.

    ### Mermaid Diagram
    ```mermaid
    graph TD
        A[Start] --> B{Is it a number?}
        B -- Yes --> C[Process the number]
        B -- No --> D[Throw an error]
        C --> E[End]
        D --> E
    ```

    ### Mathematical Expression
    $$ \varphi = \frac{1 + \sqrt{5}}{2} $$

    Inline math: $E = mc^2$

    ### Code Examples
    === "Python"
        ```python
        def add(a: int, b: int) -> int:
            return a + b
        ```

    === "JavaScript"
        ```javascript
        function add(a, b) {
            return a + b;
        }
        ```

    ??? example "Expandable Example"
        This is a collapsible content section that shows more detailed usage:

        ```python
        result = example_function_with_docstring(5, 3)
        print(f"Result: {result}")  # Output: Result: 8
        ```

    ### Table Example
    | Input A | Input B | Result |
    |---------|---------|--------|
    | 1       | 2       | 3      |
    | 5       | 7       | 12     |
    | 0       | 0       | 0      |

    #### Args:
    - `a` (*int*): The first integer.
    - `b` (*int*): The second integer.

    #### Returns:
    - *int*: The sum of the two integers.

    #### Raises:
    - `TypeError`: If inputs are not integers.

    #### Examples:
    ```python
    >>> example_function_with_docstring(2, 3)
    5
    >>> example_function_with_docstring(0, 0)
    0
    ```
    """
    return a + b


class ExampleClass:
    """An example class to demonstrate docstring formatting."""

    def __init__(self, value: int):
        """Initialize the ExampleClass with a value.

        Args:
            value (int): The initial value for the class instance.
        """
        self.value = value

    def increment(self, amount: int) -> int:
        """Increment the stored value by a specified amount.

        Args:
            amount (int): The amount to increment the value by.

        Returns:
            int: The new value after incrementing.
        """
        self.value += amount
        return self.value

    def example_method_with_docstring(self, a: int, b: int) -> int:
        # ruff: noqa: E501
        """Docstring with a cross-reference to the example function.

        This method demonstrates how to reference another function's docstring.
        It calls `example_function_with_docstring` with sample arguments.

        Cross-reference:
            1. [fatpy.examples.docstring_example_tmp.example_function_with_docstring][]
            2. [`Reference with title`][fatpy.examples.docstring_example_tmp.example_function_with_docstring]

        Args:
            a (int): The first integer value.
            b (int): The second integer value.

        Returns:
            int: The result of the example function.
        """
        return example_function_with_docstring(a, b)
