import h5py
from utils.plot import VolumeViewer, visualize_image
import numpy as np
import cupy as cp
from utils.extract_slices import extract_slices_from_points




file = r"D:\mmissana\data\4DRVQ_Jinyang\voxels\101001.h5"

def print_structure(name, obj):
    print(name, obj)


with h5py.File(file, 'r') as h5_file:
    h5_file.visititems(print_structure)
    image_0 = h5_file['Input']['grid00'][:]
    image_1 = h5_file['GroundTruth']['grid00'][:]
    volume_superimposed = image_0 + image_1 * 50
    volume = cp.asarray(volume_superimposed)

tricuspid_valve = cp.asarray([-0.019, -0.077, -0.002])
apex = cp.asarray([0.050, -0.130, -0.024])

degrees = np.linspace(np.pi, 2*np.pi, 100)
imgs = extract_slices_from_points(image_0, image_1, tricuspid_valve, apex, degrees)

viewer = VolumeViewer(imgs)
viewer.show()