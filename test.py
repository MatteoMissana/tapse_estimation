import h5py
import os
import numpy as np

# Path to one of your new files
folder_path = 'D:\mmissana\data\RV_PATIENTS\RV_patients_annotated'

count = 0
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    for file in os.listdir(subfolder_path):
        if 'interpolated' in file:
            file_path = os.path.join(subfolder_path, file)
            # Open the file in read mode
            with h5py.File(file_path, 'r') as h5_file:
                frames = h5_file['frames'][()]
                print(file_path, frames.shape[2])
                count += frames.shape[2]
print(count)