import h5py
from utils.plot import VolumeViewer, visualize_image
import numpy as np
import cupy as cp
from utils.extract_slices import extract_slices_from_points
from utils.paired_point_reg import paired_point_registration

destination_points = np.array([[194,166,159], [134,31,165], [55,129,142],
                               [131,150,162], [104,52,134],[202,84,261]])

source_points = np.array([[-0.02, -0.078, -0.003],
                           [0.015, -0.080, 0.064],
                             [0.015,-0.073, 0.054],
                             [0.001, -0.084, 0.014],
                             [0.029,-0.067,0.053],
                             [0.051, -0.131,-0.021]])

T = paired_point_registration(source_points, destination_points)
print(T)

file = r"D:\mmissana\data\4DRVQ_Jinyang\voxels\100001.h5"

def mesh_point_to_voxel(point, origin, delta):
    """
    Convert a point from mesh coordinates to voxel coordinates.
    
    Parameters:
    - point: np.array of shape (3,), the point in mesh coordinates.
    - origin: np.array of shape (3,), the origin of the voxel grid in mesh coordinates.
    - delta: np.array of shape (3,3), the transformation matrix from voxel to mesh coordinates.
    
    Returns:
    - voxel_idx: np.array of shape (3,), the voxel index corresponding to the input point.
    """
    inv_delta = np.linalg.inv(delta)  # Invert the transformation matrix
    inv_delta = np.array([[2181.818, 0,0],[0,2730.769,0],[0,0,1636.36]])
    voxel_idx = abs(inv_delta @ (point - origin)) # Apply the inverse transformation
    return np.round(voxel_idx).astype(float)

with h5py.File(file, 'r') as h5_file:
    image_0 = h5_file['Input']['grid00'][:]
    image_1 = h5_file['GroundTruth']['grid00'][:]
    vres = h5_file["VolumeInfo"]["resolution"][()]
    origin = h5_file["VolumeInfo"]["origin"][()]
    print('origin', origin)
    directions = h5_file["VolumeInfo"]["directions"][()]
    print('directions', directions)
    shape = h5_file["VolumeInfo"]["shape"][()]
    print(shape)

tricuspid_valve = np.asarray([-0.019, -0.077, 0.003]) # both points coordinates come from 3d slicer
apex = np.asarray([0.050, -0.130, -0.024])

'''
# Convert the 3D point to homogeneous coordinates by appending 1
point_homogeneous = np.append(tricuspid_valve, 1)

# Apply the transformation matrix T
transformed_point_homogeneous = T @ point_homogeneous

# Convert back to 3D coordinates by taking the first three components
# (divide by the last component if it is not 1; here it's 1)
idx_tric = transformed_point_homogeneous[:3]
print(idx_tric)

# Convert the 3D point to homogeneous coordinates by appending 1
point_homogeneous = np.append(apex, 1)

# Apply the transformation matrix T
transformed_point_homogeneous = T @ point_homogeneous
idx_apex = transformed_point_homogeneous[:3]
print(idx_apex)

idx_tric = cp.asarray(idx_tric)
idx_apex = cp.asarray(idx_apex)
'''

delta = vres * directions / np.linalg.norm(directions, axis=0)
def _get_coord(p):
        return np.linalg.inv(delta) @ (p - origin)
#print('delta', delta)

coord_tric = _get_coord(tricuspid_valve)
coord_tric = abs(coord_tric)
coord_tric[0], coord_tric[1] = coord_tric[1], coord_tric[0]
coord_tric[1] = shape[1] - coord_tric[1]

#coord_tric = abs(coord_tric)
print('coord_tric',coord_tric)

coord_apex = _get_coord(apex)
coord_apex = abs(coord_apex)
coord_apex[0], coord_apex[1] = coord_apex[1], coord_apex[0]
coord_apex[1] = shape[1] - coord_apex[1]

#coord_apex = abs(coord_apex)
print(coord_apex)


viewer = VolumeViewer(image_1*255)
viewer.clicked_points.append(coord_tric)
viewer.clicked_points.append(coord_apex)
viewer.show()


coord_tric= cp.asarray(coord_tric)
print(coord_tric)
coord_apex= cp.asarray(coord_apex)

degrees = np.linspace(np.pi, 2*np.pi, 100)
imgs = extract_slices_from_points(image_0, image_1, tric_valve=coord_tric, apex=coord_apex, degrees=degrees)

viewer = VolumeViewer(imgs)
viewer.show()