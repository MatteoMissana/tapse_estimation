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

degrees = np.linspace(np.pi, 2*np.pi, 100)
imgs = extract_slices(image_0, image_1, degrees)
viewer = VolumeViewer(imgs)
viewer.show()