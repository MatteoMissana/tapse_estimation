import h5py
import os
import numpy as np

path = r'd:\mmissana\data\RV_PATIENTS\RV_patients_annotated_renamed'

annotations = []
images = []

sum = 0
for subfolder in os.listdir(path):
    sub_path = os.path.join(path, subfolder)
    print(f"patient: {sub_path.split(os.sep)[-1]}")
    for file in os.listdir(sub_path):
        if 'interpolated' in file:
            file_path = os.path.join(sub_path, file)
            with h5py.File(file_path, 'r') as h5_file:
                print(h5_file['frames'][()].transpose((2, 0, 1)).shape[0])
                sum = sum + h5_file['frames'][()].transpose((2, 0, 1)).shape[0]

print(f"Total number of labeled images: {sum}")