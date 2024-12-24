import numpy as np
from typing import Optional, Tuple

def get_plane_normal(coefficients: np.ndarray) -> np.ndarray:
    """
    Calculate the normal vector of a plane defined by the coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficients of the plane in the form [a, b, c, d] where the plane is defined as
        a*x + b*y + c*z + d = 0.

    Returns
    -------
    np.ndarray
        The normal vector of the plane.
    """
    if len(coefficients) != 4:
        raise ValueError("The coefficients must have 4 elements.")
    if np.all(coefficients[:3] == 0):
        raise ValueError("The coefficients must not be all zeros.")
    return coefficients[:3] / np.linalg.norm(coefficients[:3])

def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Fit a plane to a set of points in 3D space.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates of the points.
    y : np.ndarray
        y-coordinates of the points.
    z : np.ndarray
        z-coordinates of the points.

    Returns
    -------
    np.ndarray
        Coefficients of the plane in the form [a, b, c, d] where the plane is defined as
        a*x + b*y + c*z + d = 0.
    """

    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("The input arrays must have the same shape.")
    

    A = np.c_[x, y, np.ones(x.shape)]
    coefficients, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    coefficients = np.insert(coefficients, 2, -1)
    return coefficients

def get_plane_dip(coefficients: np.ndarray, degrees: Optional[bool] = True, z_direction: Optional[int] = -1):
    """
    Calculate the dip of a plane defined by the coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficients of the plane in the form [a, b, c, d] where the plane is defined as
        a*x + b*y + c*z + d = 0.
    degrees : bool, optional
        If True, return the dip in degrees. Otherwise, return in radians. Default is True.

    Returns
    -------
    float
        The dip of the plane.
    """
    normal = get_plane_normal(coefficients)
    z_normal = np.array([0, 0, 1 * z_direction])
    dip = np.arccos(np.dot(normal, z_normal))
    if degrees:
        dip = np.degrees(dip)
    return dip

def get_plane_strike(coefficients: np.ndarray, degrees: Optional[bool] = True) -> float:
    """
    Calculate the strike of a plane defined by the coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficients of the plane in the form [a, b, c, d] where the plane is defined as
        a*x + b*y + c*z + d = 0.
    degrees : bool, optional
        If True, return the strike in degrees. Otherwise, return in radians. Default is True.

    Returns
    -------
    float
        The strike of the plane.
    """

    if np.dot(coefficients[:3], np.array([0, 0, 1])) == 0:
        raise ValueError("The plane is parallel to the vertical direction.")

    _, direction = get_plane_intersection(coefficients, np.array([0, 0, 1, 0]))
    north_normal = np.array([0, 1, 0])
    strike = np.arccos(np.dot(direction, north_normal))
    if degrees:
        strike = np.degrees(strike)
    return strike

def get_plane_intersection(coefficients1: np.ndarray, coefficients2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the intersection line of two planes defined by the coefficients.

    Parameters
    ----------
    coefficients1 : np.ndarray
        Coefficients of the first plane in the form [a, b, c, d] where the plane is defined as
        a*x + b*y + c*z + d = 0.
    coefficients2 : np.ndarray
        Coefficients of the second plane in the form [a, b, c, d] where the plane is defined as
        a*x + b*y + c*z + d = 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Coefficients of the intersection line in the form (point, direction) where the line is defined as
        point + t*direction.
    """
    normal1 = coefficients1[:3] / np.linalg.norm(coefficients1[:3])
    normal2 = coefficients2[:3] / np.linalg.norm(coefficients2[:3])

    point1 = np.array([0, 0, -coefficients1[3] / coefficients1[2]])
    point2 = np.array([0, 0, -coefficients2[3] / coefficients2[2]])

    normals_stacked = np.vstack((normal1, normal2))

    matrix = np.block([
        [2 * np.eye(3), normals_stacked.T],
        [normals_stacked, np.zeros((2, 2))]
    ])

    dot_a = np.dot(point1, normal1)
    dot_b = np.dot(point2, normal2)
    array_y = np.array([0, 0, 0, dot_a, dot_b])

    solution = np.linalg.solve(matrix, array_y)

    point_line = solution[:3]
    direction_line = np.cross(normal1, normal2)

    point_line = np.where(np.abs(point_line) < 1e-7, 0, point_line)
    direction_line = np.where(np.abs(direction_line) < 1e-7, 0, direction_line)

    direction_line /= np.linalg.norm(direction_line)

    return point_line, direction_line

def project_to_plane(points: np.ndarray, coefficients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project points to a plane defined by the coefficients.

    Parameters
    ----------
    points : np.ndarray
        The points to project.
    coefficients : np.ndarray
        Coefficients of the plane in the form [a, b, c, d] where the plane is defined as
        a*x + b*y + c*z + d = 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The projected points and the distances between the original points and the projected points.
    """
    normal = coefficients[:3] / np.linalg.norm(coefficients[:3])
    point_plane = np.array([0, 0, -coefficients[3] / coefficients[2]])

    distances = np.dot(points - point_plane, normal)
    projected_points = points - distances[:, np.newaxis] * normal

    return projected_points, distances

def change_coordinate_system(points: np.ndarray, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """
    Change the coordinate system of a point cloud.

    Parameters
    ----------
    points : np.ndarray
        The point cloud to transform.
    origin : np.ndarray
        The origin of the new coordinate system.
    x_axis : np.ndarray
        The x-axis of the new coordinate system.
    y_axis : np.ndarray
        The y-axis of the new coordinate system.

    Returns
    -------
    np.ndarray
        The transformed point cloud.
    """
    if points.ndim == 1:
        points = points[np.newaxis]
    if points.shape[1] != 3:
        raise ValueError("The input points must have 3 columns.")

    z_axis = np.cross(x_axis, y_axis)
    transformation_matrix = np.linalg.inv(np.vstack([x_axis, y_axis, z_axis]).T)

    transformed_points = np.dot(points - origin, transformation_matrix.T)
    return transformed_points

def plot_plane(ax, coefficients: np.ndarray, x_range: np.ndarray, y_range: np.ndarray, color: str = "b", alpha: float = 0.5, **kwargs):
    """
    Plot a plane defined by the coefficients.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot the plane on.
    coefficients : np.ndarray
        Coefficients of the plane in the form [a, b, c, d] where the plane is defined as
        a*x + b*y + c*z + d = 0.
    x_range : np.ndarray
        The range of x-values to plot the plane.
    y_range : np.ndarray
        The range of y-values to plot the plane.
    color : str, optional
        The color of the plane, by default "b".
    alpha : float, optional
        The transparency of the plane, by default 0.5.
    """
    a, b, c, d = coefficients
    x, y = np.meshgrid(x_range, y_range)
    z = -(a * x + b * y + d) / c
    ax.plot_surface(x, y, z, color=color, alpha=alpha, **kwargs)
