import os
import h5py
import numpy as np

folder_path = r'data/2d_focused_rv/RV_focused_TEE_images_annotated'

n_images = 0
for subfolder in os.listdir(folder_path):
    if subfolder == 'readme.txt':
        continue
    sub_path = os.path.join(folder_path, subfolder)
    for file in os.listdir(sub_path):
        if not 'interpolated' in file:
            # print(f"Skipping {file}, already processed.")
            continue
        file_path = os.path.join(sub_path, file)
        with h5py.File(file_path, 'r') as h5_file:
            annotations = h5_file['annotations'][()]
            if np.all(annotations == 0):
                # print(f"File {file} has no annotations.")
                continue
            else:
                print(f"{file_path}      {len(annotations)}")
                n_images += len(annotations)
print(f"Total number of images with annotations: {n_images}")