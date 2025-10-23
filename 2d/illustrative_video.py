import numpy as np
import os
import torch
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from dataloader.preprocessing import preprocess_images, apply_lut, resize_or_crop_image_np_nokeypoints
from utils.plot import save_image_ill
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from models.models import Unet


def process_h5_file_single_illustrative(
    file_path,
    model,
    device,
    save_path,
    folder,
    threshold=0.875,
):
    """
    Processes a .h5 file frame by frame (no batching).
    """

    with h5py.File(file_path, 'r') as f:
        images = f['tissue']['data'][()]  # (H, W, N)

    images = apply_lut(images.transpose(1, 0, 2)[:, ::-1, :])
    images = resize_or_crop_image_np_nokeypoints(images.transpose(2, 0, 1))
    images = images / images.max()

    N = len(images)
    coordinates_array = np.zeros((N, 3, 2))

    file_name = os.path.basename(file_path).replace(".h5", "")
    save_path = os.path.join(save_path, folder, file_name)
    os.makedirs(save_path, exist_ok=True)

    for i in range(N):
        img = images[i]
        img = preprocess_images(np.expand_dims(img, axis=0), model_type='U-Net', device=device)
        output = model(img.float().unsqueeze(0).to(device))

        for c in range(3):
            coordinates_array[i, c] = center_of_mass(output[0, c].detach(), thresh=threshold)


        pred_points = [tuple(coordinates_array[i, k]) for k in range(3)]
        ann_points = None
        bold_flag = False


        # Save both pred (red) and ann (green)
        save_image_ill(
            img[0, 0].cpu().numpy(),
            points=pred_points,
            save_folder=save_path,
            bold=bold_flag
        )

    return coordinates_array, None, None


def main():
    parser = argparse.ArgumentParser(description="Landmark prediction from .h5 files")
    parser.add_argument("--threshold", required=True, type=float, default=0.875, help="Threshold for center_of_mass")
    parser.add_argument("--save_images", action='store_true', help="Flag to save images with predicted keypoints")
    parser.add_argument('--no_sudden_movements', action='store_true', help='Flag to avoid sudden movements in keypoints')
    parser.add_argument('--threshold_sudden', type=int, default=20, help='Threshold for sudden movement detection')
    args = parser.parse_args()


    model_checkpoint = r'2d/runs/best_unet/best_model.pth'
    test_path = r'D:\mmissana\data\RV_PATIENTS\RV_patients_annotated_renamed'
    save_path = r'2d/illustrative_video'

    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Unet(depth=6, start_filts=16, num_residuals=0).to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)['model_state_dict'])
    model.eval()

    stats_list = []
    all_distances = []
    keypoint_names = ['FW annular point', 'Septal annular point', 'Apex']

    # --- Loop over patients (folders) ---
    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)
        if folder in ['100', '111', '140', '149', '160', '170', '190', '198', '199', '920']:
            for file in os.listdir(folder_path):
                if 'interpolated' in file:
                    file_path = os.path.join(folder_path, file)

                    coordinates_array, stats, distances = process_h5_file_single_illustrative(
                                                                                    file_path=file_path,
                                                                                    model=model,
                                                                                    device=device,
                                                                                    save_path=save_path,
                                                                                    folder=folder,
                                                                                    threshold=args.threshold,
                                                                                )

if __name__ == "__main__":
    main()