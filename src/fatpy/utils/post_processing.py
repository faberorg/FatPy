"""Utilities for post-processing fatigue analysis results.

This module provides functions and classes for analyzing and visualizing
fatigue analysis results, including result processing and formatting.
"""


def addition(a: float, b: float) -> float:
    r"""Add two numbers.

    This function takes two float numbers and returns their sum.

    $$ \\text{result} = a + b $$

    Args:
        a: First number to add.
        b: Second number to add.

    Returns:
        The sum of the two numbers.

    Example:
        ```python
        result = addition(2.0, 3.0)
        print(result)  # Output: 5.0
        ```

    """
    return a + b


class SomeClass:
    """A demonstration class for documentation purposes.

    This class provides examples of documented methods and properties.
    """

    def __init__(self, value: float) -> None:
        """Initialize the class with a value.

        Args:
            value: A float value to initialize the class.

        """
        self.value = value

    def multiply(self, factor: float) -> float:
        r"""Multiply the stored value by a factor.

        $$ result = self.value \\cdot factor $$

        Args:
            factor: A float factor to multiply the stored value.

        Returns:
            The product of the stored value and the factor.

        Example:
            ```python
            instance = SomeClass(5.0)
            result = instance.multiply(2.0)
            print(result)
            ```

        """
        return self.value * factor
