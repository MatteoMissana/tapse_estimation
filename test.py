import numpy as np
from utils.plot import VolumeViewer
import os

path = 'data/best_slices_2'
length = 0

for folder in os.listdir(path):
    subfolder = os.path.join(path, folder)
    for file in os.listdir(subfolder):
        if file.endswith('.npz'):
            file_path = os.path.join(subfolder, file)
            imgs_file = np.load(file_path)
            length = length + imgs_file['video'].shape[2]

print('NUMBER OF IMAGES:', length)