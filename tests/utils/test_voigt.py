import numpy as np
import pytest
from numpy.typing import NDArray

from fatpy.utils import voigt


def get_tensor(array: list[int]) -> NDArray[np.float64]:
    """Helper to create a tensor from a nested list."""
    return np.array(
        [
            [array[0], array[5], array[4]],
            [array[5], array[1], array[3]],
            [array[4], array[3], array[2]],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def vector_1d() -> NDArray[np.float64]:
    """Fixture providing a 1D Voigt vector.

    Returns:
        NDArray[np.float64]: 1D Voigt vector.
    """
    return np.arange(1, 7, dtype=np.float64)


@pytest.fixture
def vector_2d() -> NDArray[np.float64]:
    vector = np.arange(1, 13, dtype=np.float64)
    return vector.reshape((2, 6))


@pytest.fixture
def vector_3d() -> NDArray[np.float64]:
    vector = np.arange(1, 25, dtype=np.float64)
    return vector.reshape((2, 2, 6))


@pytest.fixture
def vector_4d() -> NDArray[np.float64]:
    vector = np.arange(1, 49, dtype=np.float64)
    return vector.reshape((2, 2, 2, 6))


@pytest.mark.parametrize(
    "shape,raises_value_error",
    [
        ((6,), False),
        ((2, 6), False),
        ((2, 2, 6), False),
        ((2, 2, 2, 6), False),
        ((2, 2, 2, 2, 6), False),
        ((2,), True),
        ((6, 2), True),
        ((6, 2, 2), True),
        ((6, 2, 2, 2), True),
        ((6, 2, 2, 2, 2), True),
    ],
)
def test_check_shape(shape: tuple[int, ...], raises_value_error: bool) -> None:
    """Test the shape validation function for Voigt vectors."""
    if not raises_value_error:
        voigt.check_shape(np.ones(shape, dtype=np.float64))

    else:
        with pytest.raises(ValueError):
            voigt.check_shape(np.ones(shape, dtype=np.float64))


@pytest.mark.parametrize(
    "shape",
    [
        ((6,)),
        ((2, 6)),
        ((2, 2, 6)),
        ((2, 2, 2, 6)),
        ((2, 2, 2, 2, 6)),
    ],
)
def test_voigt_to_tensor_shape(shape: tuple[int, ...]) -> None:
    """Test the shape validation function for Voigt vectors."""
    tensor = voigt.voigt_to_tensor(np.ones(shape, dtype=np.float64))
    expected_shape = shape[:-1] + (3, 3)
    assert tensor.shape == expected_shape


def test_voigt_to_tensor_1d(vector_1d: NDArray[np.float64]) -> None:
    """Test conversion from 1D Voigt vector to tensor."""
    tensor = voigt.voigt_to_tensor(vector_1d)
    expected = get_tensor(list(range(1, 7)))
    np.testing.assert_array_equal(tensor, expected)
    assert tensor.shape == (3, 3)


def test_voigt_to_tensor_2d(vector_2d: NDArray[np.float64]) -> None:
    """Test conversion from 2D Voigt vector to tensor."""
    tensor = voigt.voigt_to_tensor(vector_2d)
    expected = np.array(
        [
            get_tensor(list(range(1, 7))),
            get_tensor(list(range(7, 13))),
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(tensor, expected)
    assert tensor.shape == (2, 3, 3)


def test_voigt_to_tensor_3d(vector_3d: NDArray[np.float64]) -> None:
    """Test conversion from 3D Voigt vector to tensor."""
    tensor = voigt.voigt_to_tensor(vector_3d)
    expected = np.array(
        [
            [
                get_tensor(list(range(1, 7))),
                get_tensor(list(range(7, 13))),
            ],
            [
                get_tensor(list(range(13, 19))),
                get_tensor(list(range(19, 25))),
            ],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(tensor, expected)
    assert tensor.shape == (2, 2, 3, 3)


def test_voigt_to_tensor_4d(vector_4d: NDArray[np.float64]) -> None:
    """Test conversion from 4D Voigt vector to tensor."""
    tensor = voigt.voigt_to_tensor(vector_4d)
    expected = np.array(
        [
            [
                [
                    get_tensor(list(range(1, 7))),
                    get_tensor(list(range(7, 13))),
                ],
                [
                    get_tensor(list(range(13, 19))),
                    get_tensor(list(range(19, 25))),
                ],
            ],
            [
                [
                    get_tensor(list(range(25, 31))),
                    get_tensor(list(range(31, 37))),
                ],
                [
                    get_tensor(list(range(37, 43))),
                    get_tensor(list(range(43, 49))),
                ],
            ],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(tensor, expected)
    assert tensor.shape == (2, 2, 2, 3, 3)


def test_tensor_to_voigt_1d(vector_1d: NDArray[np.float64]) -> None:
    """Test conversion from 1D tensor to Voigt vector."""
    tensor = voigt.voigt_to_tensor(vector_1d)
    voigt_vector = voigt.tensor_to_voigt(tensor)
    np.testing.assert_array_equal(voigt_vector, vector_1d)
    assert voigt_vector.shape == (6,)


def test_tensor_to_voigt_2d(vector_2d: NDArray[np.float64]) -> None:
    """Test conversion from 2D tensor to Voigt vector."""
    tensor = voigt.voigt_to_tensor(vector_2d)
    voigt_vector = voigt.tensor_to_voigt(tensor)
    np.testing.assert_array_equal(voigt_vector, vector_2d)
    assert voigt_vector.shape == (2, 6)


def test_tensor_to_voigt_3d(vector_3d: NDArray[np.float64]) -> None:
    """Test conversion from 3D tensor to Voigt vector."""
    tensor = voigt.voigt_to_tensor(vector_3d)
    voigt_vector = voigt.tensor_to_voigt(tensor)
    np.testing.assert_array_equal(voigt_vector, vector_3d)
    assert voigt_vector.shape == (2, 2, 6)


def test_tensor_to_voigt_4d(vector_4d: NDArray[np.float64]) -> None:
    """Test conversion from 4D tensor to Voigt vector."""
    tensor = voigt.voigt_to_tensor(vector_4d)
    voigt_vector = voigt.tensor_to_voigt(tensor)
    np.testing.assert_array_equal(voigt_vector, vector_4d)
    assert voigt_vector.shape == (2, 2, 2, 6)
