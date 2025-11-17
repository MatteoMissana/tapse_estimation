import os
import h5py
import numpy as np

dataset_path = r'C:\Users\User\Desktop\final_reviewed_dataset'
count = 0


for folder in os.listdir(dataset_path):
    if folder != '106':
        folder_path = os.path.join(dataset_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with h5py.File(file_path, 'r') as f:
                data = f['tissue']['data']
                count = count + data.shape[2]

print(count)