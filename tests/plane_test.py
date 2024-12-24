import numpy as np
import pytest
from geometry3.plane import get_plane_normal, fit_plane

def test_get_plane_normal():
    # Test case 1: Normal vector of a plane with coefficients [1, 0, 0, -5]
    coefficients = np.array([1, 0, 0, -5])
    expected_normal = np.array([1, 0, 0])
    np.testing.assert_almost_equal(get_plane_normal(coefficients), expected_normal)

    # Test case 2: Normal vector of a plane with coefficients [0, 1, 0, -5]
    coefficients = np.array([0, 1, 0, -5])
    expected_normal = np.array([0, 1, 0])
    np.testing.assert_almost_equal(get_plane_normal(coefficients), expected_normal)

    # Test case 3: Normal vector of a plane with coefficients [0, 0, 1, -5]
    coefficients = np.array([0, 0, 1, -5])
    expected_normal = np.array([0, 0, 1])
    np.testing.assert_almost_equal(get_plane_normal(coefficients), expected_normal)

    # Test case 4: Normal vector of a plane with coefficients [1, 1, 1, -5]
    coefficients = np.array([1, 1, 1, -5])
    expected_normal = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
    np.testing.assert_almost_equal(get_plane_normal(coefficients), expected_normal)

    # Test case 5: Normal vector of a plane with coefficients [3, 4, 0, -5]
    coefficients = np.array([3, 4, 0, -5])
    expected_normal = np.array([3, 4, 0]) / np.linalg.norm([3, 4, 0])
    np.testing.assert_almost_equal(get_plane_normal(coefficients), expected_normal)

    # Test case 6: Normal vector of a plane with coefficients [0, 0, 0, -5]
    coefficients = np.array([0, 0, 0, -5])
    with pytest.raises(ValueError):
        get_plane_normal(coefficients)

    # Test case 7: Normal vector of a plane with coefficients [0, 0, 0]
    coefficients = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        get_plane_normal(coefficients)

def test_fit_plane():
#     # Test case 1: Fit plane to points on the plane z = 2x + 3y + 5
#     x = np.linspace(0, 1000, 1000)
#     y = np.linspace(-1000, 1000, 1000)
#     z = 2 * x + 30 * y + 5
#     expected_coefficients = np.array([2, 3, -1, 5])
#     print(fit_plane(x, y, z))
#     np.testing.assert_almost_equal(fit_plane(x, y, z), expected_coefficients, decimal=-1)

#     # Test case 2: Fit plane to points on the plane z = -x + 4y - 2
#     x = np.array([0, 1, 2])
#     y = np.array([0, 1, 2])
#     z = -x + 4 * y - 2
#     expected_coefficients = np.array([-1, 4, -1, -2])
#     np.testing.assert_almost_equal(fit_plane(x, y, z), expected_coefficients, decimal=5)

#     # Test case 3: Fit plane to points on the plane z = 3
#     x = np.array([0, 1, 2])
#     y = np.array([0, 1, 2])
#     z = np.array([3, 3, 3])
#     expected_coefficients = np.array([0, 0, -1, 3])
#     np.testing.assert_almost_equal(fit_plane(x, y, z), expected_coefficients, decimal=5)

#     # Test case 4: Fit plane to points on the plane z = 0.5x - 0.5y + 1
#     x = np.array([0, 1, 2])
#     y = np.array([0, 1, 2])
#     z = 0.5 * x - 0.5 * y + 1
#     expected_coefficients = np.array([0.5, -0.5, -1, 1])
#     np.testing.assert_almost_equal(fit_plane(x, y, z), expected_coefficients, decimal=5)

#     # Test case 5: Fit plane to points on the plane z = 0
#     x = np.array([0, 1, 2])
#     y = np.array([0, 1, 2])
#     z = np.array([0, 0, 0])
#     expected_coefficients = np.array([0, 0, -1, 0])
#     np.testing.assert_almost_equal(fit_plane(x, y, z), expected_coefficients, decimal=5)

    # Test case 6: Input arrays with different shapes
        x = np.array([0, 1])
        y = np.array([0, 1, 2])
        z = np.array([0, 1, 2])
        with pytest.raises(ValueError):
            fit_plane(x, y, z)