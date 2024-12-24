import numpy as np
import pytest
from geometry3.plane import get_plane_dip, get_plane_strike

def test_get_plane_dip():
  # Test case 1: Dip of a horizontal plane z = 5
    coefficients = np.array([0, 0, 1, -5])
    expected_dip = 180
    assert np.isclose(get_plane_dip(coefficients), expected_dip)

    # Test case 2: Dip of a vertical plane x = 5
    coefficients = np.array([1, 0, 0, -5])
    expected_dip = 90
    assert np.isclose(get_plane_dip(coefficients), expected_dip)

    # Test case 3: Dip of a plane y = 5
    coefficients = np.array([0, 1, 0, -5])
    expected_dip = 90
    assert np.isclose(get_plane_dip(coefficients), expected_dip)

    # Test case 4: Dip of a plane z = x + y + 5
    coefficients = np.array([1, 1, 1, -5])
    expected_dip = np.degrees(np.arccos(1 / np.sqrt(3)))
    assert np.isclose(get_plane_dip(coefficients, z_direction=1), expected_dip)

    # Test case 5: Dip of a plane z = 2x + 3y + 5
    coefficients = np.array([2, 3, 1, -5])
    expected_dip = np.degrees(np.arccos(1 / np.sqrt(14)))
    assert np.isclose(get_plane_dip(coefficients, z_direction=1), expected_dip)

    # Test case 6: Dip of a plane z = 0.5x - 0.5y + 1
    coefficients = np.array([0.5, -0.5, 1, -1])
    expected_dip = np.degrees(np.arccos(1 / np.sqrt(1.5)))
    assert np.isclose(get_plane_dip(coefficients, z_direction=1), expected_dip)

    # Test case 7: Dip of a plane z = 0.5x - 0.5y + 1 with z_direction=-1
    coefficients = np.array([0.5, -0.5, 1, -1])
    expected_dip = np.degrees(np.arccos(-1 / np.sqrt(1.5)))
    assert np.isclose(get_plane_dip(coefficients, z_direction=-1), expected_dip)


def test_get_plane_strike():
    # Test case 1: Strike of a plane with coefficients [1, 0, 0, -5]
    coefficients = np.array([1, 0, 0, -5])
    with pytest.raises(ValueError):
        get_plane_strike(coefficients)

    # Test case 2: Strike of a plane with coefficients [0, 1, 0, -5]
    coefficients = np.array([0, 1, 0, -5])
    with pytest.raises(ValueError):
        get_plane_strike(coefficients)

    # Test case 3: Strike of a plane with coefficients [1, 1, 0, -5]
    coefficients = np.array([1, 1, 0, -5])
    with pytest.raises(ValueError):
        get_plane_strike(coefficients)

    # Test case 4: Strike of a plane with coefficients [1, 1, 1, -5]
    coefficients = np.array([1, 1, 1, -5])
    expected_strike = 135
    assert np.isclose(get_plane_strike(coefficients), expected_strike)

    # Test case 5: Strike of a plane with coefficients [1, 1, -1, -5]
    coefficients = np.array([1, 1, -1, -5])
    expected_strike = 135
    assert np.isclose(get_plane_strike(coefficients), expected_strike)

    # Test case 6: Strike of a plane with coefficients [-1, 1, 1, -5]
    coefficients = np.array([-1, 1, 1, -5])
    expected_strike = 45
    assert np.isclose(get_plane_strike(coefficients), expected_strike)