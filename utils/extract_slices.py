import numpy as np
from scipy.ndimage import map_coordinates
import cupy as cp
from utils.plot import VolumeViewer
from cupyx.scipy.ndimage import rotate

print(cp.cuda.Device(0).compute_capability)  # Mostra la Compute Capability
print(cp.cuda.Device(0).attributes)  # Mostra varie info sulla GPU


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

def signed_angle_between_vectors(vec, target=np.array([0, 0, 1]), ref_axis=None):
    """
    Compute the signed angle (in degrees) between `vec` and `target`.

    :param vec: Source vector (does not need to be normalized)
    :param target: Target vector (default: [0, 0, 1]), does not need to be normalized
    :param ref_axis: Optional reference axis to determine sign of rotation (default: cross product of vec and target)
    :return: Signed angle in degrees
    """
    vec = vec / np.linalg.norm(vec)  # Normalize vector
    target = target / np.linalg.norm(target)

    v_cross = np.cross(vec, target)  # Axis of rotation
    dot_product = np.dot(vec, target)  # Cosine of angle
    angle_rad = np.arctan2(np.linalg.norm(v_cross), dot_product)  # Angle in radians

    if ref_axis is None:
        ref_axis = v_cross  # Use cross product to determine sign if no reference axis is provided

    # Determine sign of the angle using dot product with reference axis
    sign = np.sign(np.dot(ref_axis, v_cross))  # +1 or -1
    angle_deg = np.degrees(angle_rad) * sign  # Convert to degrees with sign

    return angle_deg

import cupy as cp

def signed_angle_between_vectors_gpu(vec, target=cp.array([0, 0, 1]), ref_axis=None):
    """
    Compute the signed angle (in degrees) between `vec` and `target`.

    :param vec: Source vector (does not need to be normalized)
    :param target: Target vector (default: [0, 0, 1]), does not need to be normalized
    :param ref_axis: Optional reference axis to determine sign of rotation (default: cross product of vec and target)
    :return: Signed angle in degrees
    """
    vec = vec / cp.linalg.norm(vec)  # Normalize vector
    target = target / cp.linalg.norm(target)

    v_cross = cp.cross(vec, target)  # Axis of rotation
    dot_product = cp.dot(vec, target)  # Cosine of angle
    angle_rad = cp.arctan2(cp.linalg.norm(v_cross), dot_product)  # Angle in radians

    if ref_axis is None:
        ref_axis = v_cross  # Use cross product to determine sign if no reference axis is provided

    # Determine sign of the angle using dot product with reference axis
    sign = cp.sign(cp.dot(ref_axis, v_cross))  # +1 or -1
    angle_deg = cp.degrees(angle_rad) * sign  # Convert to degrees with sign

    return angle_deg


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



def slice_volume_z(vol, theta):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    # unit vector
    e_xy = [[1.0, 0.0, 0.0], [0.0, np.cos(theta), np.sin(theta)]]

    center = cp.array([vol.shape[0] / 2, vol.shape[1] / 2, vol.shape[2] / 2], dtype=float)

    e_xy = cp.array(e_xy)
    e_xy = e_xy / cp.linalg.norm(e_xy, axis=1)[:, cp.newaxis]

    # free gpu mem
    # R: rotation matrix: data_coords = center + r @ slice_coords
    ez = cp.cross(e_xy[0], e_xy[1])
    r = cp.array([e_xy[0], e_xy[1], ez], dtype=cp.float32).T

    # free gpu mem
    del e_xy, ez

    # setup slice points P with coordinates (X, Y, 0)
    mx, my = int(vol.shape[0]), int(vol.shape[1])
    xs = cp.arange(0.5 - mx / 2, 0.5 + mx / 2)
    ys = cp.arange(0.5 - my / 2, 0.5 + my / 2)

    pp = cp.zeros((3, mx, my), dtype=cp.float32)
    pp[0, :, :] = xs.reshape(mx, 1)
    pp[1, :, :] = ys.reshape(1, my)

    # Transform to data coordinates (x, y, z) - idx.shape == (3, mx, mx)
    # pure numpy solution with nearest-neighbor interpolation
    idx = cp.einsum('il,ljk->ijk', r, pp) + (0.5 + center.reshape(3, 1, 1))

    # free gpu mem
    del r, xs, ys, pp, center

    idx = idx.astype(cp.int16)
    # Find out which coordinates are out of range - shape (mx, mx)
    offpoints = cp.any(cp.array([idx[0, :, :] < 0,
                                 idx[0, :, :] >= vol.shape[0],
                                 idx[1, :, :] < 0,
                                 idx[1, :, :] >= vol.shape[1],
                                 idx[2, :, :] < 0,
                                 idx[2, :, :] >= vol.shape[2]
                                 ]), axis=0)

    idx[:, offpoints] = 0
    img = vol[idx[0], idx[1], idx[2]]

    # free gpu mem
    del offpoints, idx, vol

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    img = cp.rot90(img)
    return img


def center_volume(volume, target_x, target_y):
    '''
    pads the volume by adding 0 along the x and y axes so that the target x and y are in the center
    volume: a cupy volume 
    int: target x, target y: the coordinates you want to center'''
    X, Y, Z = volume.shape

    pad_x = 2*int((target_x - X / 2))
    pad_y = 2*int((target_y - Y / 2))


    pad_x_before = max(0, 0-pad_x)
    pad_x_after = max(0, pad_x)

    pad_y_before = max(0, 0-pad_y)
    pad_y_after = max(0, pad_y)

    padded_volume = cp.pad(volume, 
                            ((pad_x_before, pad_x_after), 
                             (pad_y_before, pad_y_after), 
                             (0, 0)), 
                            mode='constant', constant_values=0)

    return padded_volume

def crop_black_borders(volume):
    """
    Crops the black (zero-valued) borders from the volume along the x and y dimensions using CuPy.
    
    Parameters:
        volume (cupy.ndarray): A 3D CuPy array of shape (x, y, z).
    
    Returns:
        cupy.ndarray: The cropped volume with borders removed along x and y.
    """
    # Compute a 2D mask to check where at least one voxel along the z-axis is non-zero
    mask = volume.any(axis=2)

    # Get the non-zero indices along x and y
    x_nonzero = cp.where(mask.any(axis=1))[0]
    y_nonzero = cp.where(mask.any(axis=0))[0]
    
    # If the entire volume is zero, return an empty volume
    if x_nonzero.size == 0 or y_nonzero.size == 0:
        return volume[0:0, 0:0, :]
    
    # Get min and max bounds for x and y
    x_min, x_max = x_nonzero[0], x_nonzero[-1]
    y_min, y_max = y_nonzero[0], y_nonzero[-1]

    # Crop the volume using the computed bounds
    cropped_volume = volume[x_min:x_max+1, y_min:y_max+1, :]
    
    return cropped_volume


def extract_slices(input, ground_truth, degrees=np.linspace(0, 2*np.pi, 10)):
    '''
    Extracts 2D slices from a 3D medical volume after aligning the right ventricle along the z-axis.

    This function performs three main steps:
    1. **Alignment**: The volume is rotated in the YZ and XZ planes to ensure the ventricle is aligned with the z-axis.
    2. **Centering**: The user selects the tricuspid valve center in the XY-plane, ensuring slices pass through this point.
    3. **Slice Extraction**: Slices are taken with planes parallel to the z-axis, rotated at the specified degrees.

    Parameters:
    - input (numpy array): The 3D volume containing the medical image.
    - ground_truth (numpy array): The 3D segmentation of the right ventricle.
    - degrees (numpy array): An array containing rotation angles for extracting slices.

    Returns:
    - A 3D CuPy array containing the extracted slices.
    '''

    # Superimpose the segmentation onto the input volume for visualization
    volume_superimposed = input + ground_truth * 50

    # Convert to CuPy arrays for GPU processing
    volume_superimposed = cp.asarray(volume_superimposed)
    volume = cp.asarray(input)

    # Step 1: Align the volume in the YZ-plane
    viewer = VolumeViewer(volume_superimposed)
    viewer.show()
    alpha = signed_angle_between_vectors_gpu(viewer.unit_vectors[0])
    
    # Rotate to align along the z-axis
    volume_superimposed = rotate(volume_superimposed, alpha, axes=(1,2), reshape=True, 
                                 order=3, mode='constant', cval=0.0, prefilter=True)
    volume = rotate(volume, alpha, axes=(1,2), reshape=True, 
                    order=3, mode='constant', cval=0.0, prefilter=True)

    # Step 2: Align the volume in the XZ-plane
    viewer = VolumeViewer(volume_superimposed)
    viewer.show()
    alpha = signed_angle_between_vectors_gpu(viewer.unit_vectors[0])
    
    # Rotate to finalize alignment
    volume_superimposed = rotate(volume_superimposed, -alpha, axes=(0,2), reshape=True, 
                                 order=3, mode='constant', cval=0.0, prefilter=True)
    volume = rotate(volume, -alpha, axes=(0,2), reshape=True, 
                    order=3, mode='constant', cval=0.0, prefilter=True)

    # Step 3: Select the tricuspid valve center in the XY-plane
    viewer = VolumeViewer(volume_superimposed)
    viewer.show()
    target_x = viewer.clicked_points[0][1]
    target_y = viewer.clicked_points[0][0]

    # Center the volume based on the selected point
    volume = center_volume(volume_superimposed, target_x, target_y)

    # Step 4: Extract slices passing through the selected point, aligned with the z-axis
    first_slice = slice_volume_z(volume, degrees[0])
    height, width = first_slice.shape  # Get dimensions of a single slice

    # Initialize an empty CuPy array for the slices
    imgs = cp.zeros((len(degrees), height, width), dtype=first_slice.dtype)

    # Extract and store each slice
    for i, angle in enumerate(degrees):
        imgs[i] = slice_volume_z(volume, angle)
    
    imgs = crop_black_borders(imgs.transpose(1,2,0))

    return imgs  # Return the extracted slices


