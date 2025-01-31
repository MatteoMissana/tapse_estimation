import numpy as np
from scipy.ndimage import map_coordinates

def extract_planes(volume, line_point, line_direction, angles):
    """
    Extracts 2D images from a 3D volume for planes containing a given line,
    and rotates the plane around the line's direction at specific angle intervals.

    Parameters:
        volume (numpy.ndarray): 3D array of shape (322, 322, 300).
        line_point (tuple): Point on the line, given as (x, y, z).
        line_direction (tuple): Direction of the line, given as (dx, dy, dz).
        angles (list or numpy.ndarray): List of angles (in degrees) between the planes to extract.

    Returns:
        list: List of 2D arrays, each representing an extracted image.
    """
    extracted_images = []

    # Normalize the direction of the line
    line_direction = np.array(line_direction) / np.linalg.norm(line_direction)

    # Find a random perpendicular vector to the line
    perpendicular_vector = find_perpendicular_vector(line_direction)

    # Define the initial plane's basis: two perpendicular vectors to the line
    plane_u = perpendicular_vector  # First vector on the plane

    # Normalize the basis vectors
    plane_u /= np.linalg.norm(plane_u)

    # Define a grid for the initial plane
    size = int(max(volume.shape)*np.sqrt(2))

    for angle in angles:
        # Rotate the plane basis vectors around the line direction
        angle_rad = np.deg2rad(angle)
        rotation_matrix = get_rotation_matrix(line_direction, angle_rad)

        rotated_u = rotation_matrix @ plane_u

        interpolated_values = extract_slice(volume, line_point, rotated_u, size=size)

        extracted_images.append(interpolated_values)

    return extracted_images


def find_perpendicular_vector(direction):
    """
    Finds a random vector perpendicular to the given direction vector.

    Parameters:
        direction (numpy.ndarray): Direction vector (dx, dy, dz).

    Returns:
        numpy.ndarray: A vector perpendicular to the given direction.
    """
    direction = np.array(direction)
    if direction[2] != 0:  # General case
        perpendicular = np.array([1, 1, -(direction[0] + direction[1]) / direction[2]])
    elif direction[1] != 0:  # If dz == 0 but dy != 0
        perpendicular = np.array([1, -(direction[0] + direction[2]) / direction[1], 1])
    else:  # If dz == 0 and dy == 0 (dx != 0)
        perpendicular = np.array([0, 1, 0])
    return perpendicular / np.linalg.norm(perpendicular)


def get_rotation_matrix(axis, angle):
    """
    Computes the 3D rotation matrix around an axis.

    Parameters:
        axis (numpy.ndarray): Rotation axis (3,).
        angle (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    ux, uy, uz = axis

    rotation_matrix = np.array([
        [cos_a + ux ** 2 * (1 - cos_a), ux * uy * (1 - cos_a) - uz * sin_a, ux * uz * (1 - cos_a) + uy * sin_a],
        [uy * ux * (1 - cos_a) + uz * sin_a, cos_a + uy ** 2 * (1 - cos_a), uy * uz * (1 - cos_a) - ux * sin_a],
        [uz * ux * (1 - cos_a) - uy * sin_a, uz * uy * (1 - cos_a) + ux * sin_a, cos_a + uz ** 2 * (1 - cos_a)]
    ])

    return rotation_matrix

def extract_slice(volume, plane_point, plane_normal, size=322):
    """
    Extracts a slice from a 3D volume corresponding to a specified plane.

    Parameters:
        volume (numpy.ndarray): 3D array representing the volume.
        plane_point (tuple): A point on the plane, given as (x, y, z).
        plane_normal (tuple): The normal vector of the plane, given as (nx, ny, nz).
        size (int): The size of the output slice (default is 322).

    Returns:
        numpy.ndarray: 2D array representing the extracted slice.
    """
    # Normalize the plane normal
    plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)

    # Find two orthogonal vectors in the plane
    # Start by finding one vector perpendicular to the normal
    perp_vector1 = find_perpendicular_vector(plane_normal)

    # Find the second vector using a cross product
    perp_vector2 = np.cross(plane_normal, perp_vector1)

    # Normalize the two vectors
    perp_vector1 /= np.linalg.norm(perp_vector1)
    perp_vector2 /= np.linalg.norm(perp_vector2)

    # Create a grid in the plane using the two perpendicular vectors
    grid_u, grid_v = np.meshgrid(
        np.linspace(-size / 2, size / 2, size),
        np.linspace(-size / 2, size / 2, size)
    )

    # Compute the coordinates of the grid points in 3D space
    grid_points = (
        plane_point[0] + grid_u * perp_vector1[0] + grid_v * perp_vector2[0],
        plane_point[1] + grid_u * perp_vector1[1] + grid_v * perp_vector2[1],
        plane_point[2] + grid_u * perp_vector1[2] + grid_v * perp_vector2[2]
    )

    # Interpolate the values of the volume at the grid points
    slice_values = map_coordinates(
        volume,
        [grid_points[0].ravel(),
         grid_points[1].ravel(),
         grid_points[2].ravel()],
        order=1,
        mode='constant',
        cval=0
    )

    # Reshape the result into a 2D array
    return slice_values.reshape(size, size)


def rotation_matrix_from_vectors(vec, target=np.array([0, 0, 1])):
    """
    Compute the rotation matrix that aligns `vec` with the `target` vector.
    Uses Rodrigues' rotation formula.

    :param vec: Source vector (must be a unit vector)
    :param target: Target vector (default: [0, 0, 1])
    :return: 3x3 rotation matrix
    """
    vec = vec / np.linalg.norm(vec)  # Ensure it's a unit vector
    target = target / np.linalg.norm(target)

    v = np.cross(vec, target)  # Rotation axis
    print(v)
    c = np.dot(vec, target)  # Cosine of the angle
    s = np.linalg.norm(v)  # Sine of the angle

    if s == 0:  # Already aligned
        return np.eye(3)

    # Skew-symmetric cross-product matrix of v
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
    return R


def rotate_volume(volume, R):
    """
    Apply a 3D rotation matrix to a volume.

    :param volume: 3D NumPy array (H, W, D)
    :param R: 3x3 rotation matrix
    :return: Rotated volume
    """
    coords = np.array(np.meshgrid(
        np.arange(volume.shape[0]),
        np.arange(volume.shape[1]),
        np.arange(volume.shape[2]),
        indexing='ij'
    )).reshape(3, -1)  # (3, N) coordinate grid

    # Rotate coordinates
    new_coords = R @ (coords - np.array(volume.shape)[:, None] / 2) + np.array(volume.shape)[:, None] / 2

    # Interpolate rotated volume
    rotated_volume = map_coordinates(volume, new_coords, order=1, mode='nearest')

    return rotated_volume.reshape(volume.shape)

