import h5py
from utils.plot import view_volume, visualize_image
import numpy as np
from utils.extract_slices import extract_planes


file = r"C:\Users\User\Desktop\uni_matteo\quinto_anno\tesi_magistrale\data\4DRVQ_Jinyang\voxels\100001.h5"

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
image_superimposed = image_superimposed.transpose(2,0,1)
print(image_superimposed.max())
view_volume(image_superimposed)


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
