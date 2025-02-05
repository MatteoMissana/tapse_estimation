import h5py
from utils.plot import VolumeViewer, visualize_image
import numpy as np
import cupy as cp
from utils.extract_slices import extract_slices
from cupyx.scipy.ndimage import rotate



file = r"D:\mmissana\data\4DRVQ_Jinyang\voxels\101001.h5"

def print_structure(name, obj):
    print(name, obj)


with h5py.File(file, 'r') as h5_file:
    h5_file.visititems(print_structure)
    image_0 = h5_file['Input']['grid00'][:]
    image_1 = h5_file['GroundTruth']['grid00'][:]

degrees = np.linspace(0, 2*np.pi, 10)
imgs = extract_slices(image_0, image_1, degrees)
imgs = imgs.transpose(1, 2, 0)
viewer = VolumeViewer(imgs)
viewer.show()

'''
image_superimposed = image_0 + image_1*50

volume_superimposed = cp.asarray(image_superimposed)
volume = cp.asarray(image_0)

viewer = VolumeViewer(volume_superimposed)
viewer.show()

alpha = signed_angle_between_vectors_gpu(viewer.unit_vectors[0])
volume_superimposed = rotate(volume_superimposed, alpha, axes=(1,2), reshape=True, order=3, mode='constant', cval=0.0, prefilter=True)
volume = rotate(volume, alpha, axes=(1,2), reshape=True, order=3, mode='constant', cval=0.0, prefilter=True)


viewer = VolumeViewer(volume_superimposed)
viewer.show()
alpha = signed_angle_between_vectors_gpu(viewer.unit_vectors[0])
volume_superimposed = rotate(volume_superimposed, alpha, axes=(1,2), reshape=True, order=3, mode='constant', cval=0.0, prefilter=True)
volume = rotate(volume, -alpha, axes=(0,2), reshape=True, order=3, mode='constant', cval=0.0, prefilter=True)


viewer = VolumeViewer(volume_superimposed)
viewer.show()
target_x = viewer.clicked_points[0][1]
target_y = viewer.clicked_points[0][0]
volume = center_volume(volume_superimposed, target_x, target_y)


degrees = np.linspace(0, 2*np.pi, 10)
for angle in degrees:
    img = slice_volume_z(volume, angle)
    visualize_image(img)

'''