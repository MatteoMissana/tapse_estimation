import os
import h5py
import numpy as np

dataset_path = r'C:\Users\User\Desktop\final_reviewed_dataset'
count = 0


for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        print(file_path)
        with h5py.File(file_path, 'r') as f:
            data = f['tissue']['data']
            print(data.shape[2])
            ann = f['annotations']
            print(ann.shape)
            count = count + ann.shape[0]

print(count)