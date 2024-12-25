# Geometry3

Geometry3 is a personal Python library for performing 3D geometry calculations. This module focuses on providing utility functions for plane calculations using pure NumPy operations without any additional dependencies or constructions. The library is intended for educational and experimental purposes and is not optimized for performance.

## Features

- Calculate the normal vector of a plane
- Fit a plane to a set of points in 3D space
- Calculate the dip and strike of a plane
- Calculate the intersection line of two planes
- Project points to a plane
- Change the coordinate system of a point cloud
- Plot a plane using Matplotlib

## Installation

To install the required dependencies, use Poetry:

```sh
poetry install
```

## Usage

### Calculate Plane Normal

```python
import numpy as np
from geometry3.plane import get_plane_normal

coefficients = np.array([1, 0, 0, -5])
normal = get_plane_normal(coefficients)
```

### Fit Plane to Points

```python
import numpy as np
from geometry3.plane import fit_plane

x = np.array([0, 1, 2])
y = np.array([0, 1, 2])
z = np.array([3, 3, 3])
coefficients = fit_plane(x, y, z)
```

### Calculate Plane Dip

```python
import numpy as np
from geometry3.plane import get_plane_dip

coefficients = np.array([1, 1, 1, -5])
dip = get_plane_dip(coefficients)
print(dip)
```

### Calculate Plane Strike

```python
import numpy as np
from geometry3.plane import get_plane_strike

coefficients = np.array([1, 1, 1, -5])
strike = get_plane_strike(coefficients)
```

- Project Points to Plane

```python
import numpy as np
from geometry3.plane import project_to_plane

points = np.array([[1, 2, 3], [4, 5, 6]])
coefficients = np.array([1, 0, 0, -5])
projected_points, distances = project_to_plane(points, coefficients)
```

- Change Coordinate System

```python
import numpy as np
from geometry3.plane import change_coordinate_system

points = np.array([[1, 2, 3], [4, 5, 6]])
origin = np.array([0, 0, 0])
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
transformed_points = change_coordinate_system(points, origin, x_axis, y_axis)
```

- Plot Plane

```python
import numpy as np
import matplotlib.pyplot as plt
from geometry3.plane import plot_plane

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
coefficients = np.array([1, 1, 1, -5])
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
plot_plane(ax, coefficients, x_range, y_range)
plt.show()
```

## Running Tests

To run the tests, use the following command:

```sh
poetry run pytest
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

This library is a personal project and is intended for educational and experimental purposes. It leverages pure NumPy calculations without any additional classes to keep the implementation simple and straightforward.