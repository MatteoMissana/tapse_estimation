import h5py
from utils.plot import VolumeViewer, visualize_image
import numpy as np
import cupy as cp
from utils.extract_slices import extract_slices_from_points


file = r"D:\mmissana\data\4DRVQ_Jinyang\voxels\101001.h5"

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
    voxel_idx = inv_delta @ (point - origin)  # Apply the inverse transformation
    return voxel_idx#np.round(voxel_idx).astype(int)  # Round to nearest voxel index

with h5py.File(file, 'r') as h5_file:
    image_0 = h5_file['Input']['grid00'][:]
    image_1 = h5_file['GroundTruth']['grid00'][:]
    vres = h5_file["VolumeInfo"]["resolution"][()]
    origin = h5_file["VolumeInfo"]["origin"][()]
    directions = h5_file["VolumeInfo"]["directions"][()]
    shape = h5_file["VolumeInfo"]["shape"][()]
    print(shape)

tricuspid_valve = np.asarray([-0.019, -0.077, -0.004]) # both points coordinates come from 3d slicer
apex = np.asarray([0.050, -0.131, -0.023])

delta = vres * directions / np.linalg.norm(directions, axis=0)

coord_tric = abs(mesh_point_to_voxel(tricuspid_valve, origin, delta))
print(coord_tric)
coord_apex = abs(mesh_point_to_voxel(apex, origin, delta))
print(coord_apex)


coord_tric= cp.asarray(coord_tric)
print(coord_tric)
coord_apex= cp.asarray(coord_apex)

degrees = np.linspace(np.pi, 2*np.pi, 100)
imgs = extract_slices_from_points(image_0, image_1, tric_valve=coord_tric, apex=coord_apex, degrees=degrees)

viewer = VolumeViewer(imgs)
viewer.show()