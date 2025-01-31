import h5py
from utils.plot import VolumeViewer, visualize_image
import numpy as np
from utils.extract_slices import extract_planes, rotation_matrix_from_vectors, rotate_volume, signed_angle_between_vectors
import scipy


file = r"D:\mmissana\data\4DRVQ_Jinyang\voxels\100001.h5"

def create_parallelepiped(shape, start, size):
    """
    Create a 3D NumPy array with a parallelepiped of 1s inside a volume of 0s.

    :param shape: Tuple (H, W, D) -> Shape of the full volume
    :param start: Tuple (x, y, z) -> Starting position of the parallelepiped
    :param size: Tuple (dx, dy, dz) -> Size of the parallelepiped
    :return: 3D NumPy array with the parallelepiped
    """
    volume = np.zeros(shape, dtype=np.uint8)  # Create a 3D volume filled with 0s

    x_start, y_start, z_start = start
    dx, dy, dz = size

    # Set 1s in the parallelepiped region
    volume[x_start:x_start + dx, y_start:y_start + dy, z_start:z_start + dz] = 255

    return volume


# Create a volume of shape (100, 100, 100) with a parallelepiped at (30,30,30) of size (20,40,10)
volume = create_parallelepiped(shape=(100, 100, 100), start=(30, 30, 30), size=(20, 40, 10))

def print_structure(name, obj):
    print(name, obj)


with h5py.File(file, 'r') as h5_file:
    h5_file.visititems(print_structure)
    image_0 = h5_file['Input']['grid00'][:]
    image_1 = h5_file['GroundTruth']['grid00'][:]



#view_volume(image_0)

print(image_1.shape)
print(image_1.max())
print(np.where(image_1[:,:,200]==True))
image_superimposed = image_0 + image_1*50
print(image_superimposed.max())
print(image_superimposed.max())
volume = image_superimposed

viewer = VolumeViewer(image_superimposed)
viewer.show()
print(viewer.clicked_points)
print(viewer.unit_vectors)
alpha = signed_angle_between_vectors(viewer.unit_vectors[0])
volume = scipy.ndimage.rotate(volume, alpha, axes=(1,2), reshape=True, order=3, mode='constant', cval=0.0, prefilter=True)


viewer = VolumeViewer(volume)
viewer.show()
print(viewer.clicked_points)
print(viewer.unit_vectors)
alpha = signed_angle_between_vectors(viewer.unit_vectors[0])
volume = scipy.ndimage.rotate(volume, alpha, axes=(0,2), reshape=True, order=3, mode='constant', cval=0.0, prefilter=True)


viewer = VolumeViewer(volume)
viewer.show()

'''
image_final = image_0+mask
image_final = image_final.transpose(0,2,1)
line_point = (151, 151, 150)
line_direction = (0, 0, 1)
angles = np.linspace(0, 360, 36)  # 36 planes, one every 10 degrees

planes = extract_planes(mask, line_point, line_direction, angles)

planes = np.asarray(planes)
planes = planes.transpose(1,2,0)
visualize_image(image_1[:,:, 100])
'''
