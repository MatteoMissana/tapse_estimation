import numpy as np
from utils.results_visualization import create_video
import cupy as cp
import os

'''allows to create a video given some frames'''

folder_path = r"D:\mmissana\data\best_slices"
save_path = r"D:\mmissana\data\best_slices_videos"

if not os.path.exists(save_path):
    os.makedirs(save_path)

for subfolder in os.listdir(folder_path):
    folder = os.path.join(folder_path, subfolder)
    npz_path = os.path.join(folder, 'video_best_slice.npz')

    # **Check if file exists before loading**
    if not os.path.exists(npz_path):
        print(f"Skipping {subfolder}, video_best_slice.npz not found.")
        continue

    # **Load using NumPy**
    imgs_file = np.load(npz_path)
    imgs = imgs_file['video']  # Already in NumPy format

    # **Save video from NumPy array**
    output_path = os.path.join(save_path, f"{subfolder}.mp4")
    create_video(imgs, output_path, fps = 5)  # 5 frames per second