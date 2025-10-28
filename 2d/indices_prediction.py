import os
import h5py
import torch
import numpy as np
import pandas as pd
import argparse

from dataloader.preprocessing import preprocess_images, apply_lut, resize_or_crop_image_np_nokeypoints
from models.models import Unet
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from postprocessing.indices_calculation import tric_apex_distance_calculation, tapse_calculation, find_parallel_direction
from postprocessing.kalman_filter import KalmanFilter
from postprocessing.pan_tompkins import pan_tompkins_detector
from utils.plot import visualize_image, save_image

"""
This script processes cardiac ultrasound sequences stored in HDF5 files to estimate 
right ventricular function indices such as TAPSE, RVLSF, and RVFAC. It uses a trained 
U-Net segmentation model to detect anatomical landmarks, applies optional Kalman filtering 
for smoothing, calculates indices for each heartbeat, and stores the results in an Excel file. 
Model parameters and processing options can be customized via command-line arguments.
"""

def predict_indices(model, 
                    test_path, 
                    apply_filter=True, 
                    device='cpu',
                    tapse_calc='distance', 
                    reduction='max', 
                    patient=None, 
                    save_images=False, 
                    images_path=None, 
                    threshold=0.8, 
                    two_dimensional=True, 
                    count_beats=False,
                    area_method = 'triangle',
                    best_combination = False,
                    no_sudden_movements = False,
                    threshold_sudden = 20,
                    best_rvlsffw = False
                    ):
    """Compute indices from a cardiac HDF5 sequence using a trained segmentation model."""

    # Load ultrasound + ECG data
    with h5py.File(test_path, 'r') as f:
        images = f['tissue']['data'][()]
        images_times = f['tissue']['times'][()]
        ecg = f['ecg']['ecg_data'][()]
        ecg_times = f['ecg']['ecg_times'][()]
        pixelsize = f['tissue']['pixelsize'][()]

    fs = 1 / (ecg_times[1] - ecg_times[0])  # ECG sampling frequency
    dt = images_times[1] - images_times[0]  # Image frame interval

    # Reorient and normalize images
    if two_dimensional:
        images = apply_lut(images.transpose(1, 0, 2)[:, ::-1, :])
        images = resize_or_crop_image_np_nokeypoints(images.transpose(2, 0, 1))
        images = images / images.max()
    else: # three dimensional
        images = resize_or_crop_image_np_nokeypoints(images.transpose(2, 0, 1))
        images = images / images.max()


    # Detect R-peaks in ECG
    r_peaks = pan_tompkins_detector(ecg, fs, plot=False)
    if count_beats:
        print(f"Aquisition of patient {patient} has {len(r_peaks)-1} detected heartbeats.")
    beat_start = [np.argmin(np.abs(images_times - ecg_times[r])) for r in r_peaks]

    # Inference loop
    coordinates_array = np.zeros((len(images), 3, 2))
    for i, im in enumerate(images):
        im = preprocess_images(np.expand_dims(im, axis=0), model_type='U-Net', device=device)
        output = model(im.float().unsqueeze(0).to(device))

        coords = [center_of_mass(output[0, c].detach(), thresh=threshold) for c in range(3)]
        for j in range(3):
            coordinates_array[i, j] = coords[j]

        # visualize_image(im[0, 0].cpu().numpy(), coords)
        
        # Save images with keypoints if required
        if save_images:
            save_images_path = os.path.join(images_path, str(patient))
            os.makedirs(save_images_path, exist_ok=True)
            save_image(im[0, 0].cpu().numpy(), points=coords, save_folder=save_images_path,  bold = True)
        
    if no_sudden_movements: # check if a prediction is too far from the previous and the next one, if so, replace it with the mean of the two
        for j in range(3):
            for i in range(1, len(coordinates_array)-1):
                if np.linalg.norm(coordinates_array[i, j] - coordinates_array[i-1, j]) > threshold_sudden and np.linalg.norm(coordinates_array[i, j] - coordinates_array[i+1, j]) > threshold_sudden: 
                    coordinates_array[i, j] = (coordinates_array[i-1, j] + coordinates_array[i+1, j]) / 2


    if best_combination:
        filtered_array = coordinates_array.copy()

        # Apply Kalman filter if needed
        kfs = [KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=0.1, y_std_meas=0.1) for _ in range(3)]
        for j in range(3):
            kfs[j].x = np.matrix([coordinates_array[0, j, 0], coordinates_array[0, j, 1], 0, 0]).T

        for i, coords in enumerate(coordinates_array):
            for j in range(3):
                kfs[j].predict()
                filtered_array[i, j] = kfs[j].update(np.matrix(coords[j]).T).A1
    elif apply_filter:
        filtered_array = coordinates_array.copy()

        # Apply Kalman filter if needed
        kfs = [KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=0.1, y_std_meas=0.1) for _ in range(3)]
        for j in range(3):
            kfs[j].x = np.matrix([coordinates_array[0, j, 0], coordinates_array[0, j, 1], 0, 0]).T

        for i, coords in enumerate(coordinates_array):
            for j in range(3):
                kfs[j].predict()
                filtered_array[i, j] = kfs[j].update(np.matrix(coords[j]).T).A1
    else: # no filtering 
        filtered_array = coordinates_array.copy()


    # Rescale to physical units
    filtered_array[:, :, 0] *= pixelsize[0]
    filtered_array[:, :, 1] *= pixelsize[1]

    coordinates_array[:, :, 0] *= pixelsize[0]
    coordinates_array[:, :, 1] *= pixelsize[1]

    # Estimate direction for TAPSE
    direction = find_parallel_direction(coordinates_array[:, 0]) + find_parallel_direction(coordinates_array[:, 1])
    direction /= np.linalg.norm(direction)

    # Compute metrics for each beat
    index_container = np.zeros((len(beat_start) - 1, 17))
    for i in range(len(beat_start) - 1):

        # select frames for the current heartbeat
        window = filtered_array[beat_start[i]:beat_start[i + 1]] # select frames for the current heartbea
        window_unfiltered = coordinates_array[beat_start[i]:beat_start[i + 1]]

        # calculate shape related indexes
        (rvfac,
        diast_area, 
        syst_area, 
        rvldfw, 
        rvldsep, 
        rvlsfw, 
        rvlssep,
        rvldmid, 
        rvlsmid, 
        tadd, 
        tasd, 
        rvlsffw, 
        rvlsfsep, 
        rvlsfmid, 
        rvlsfglobal) = tric_apex_distance_calculation(window[:, 0], 
                                                        window[:, 1], 
                                                        window[:, 2], 
                                                        window_unfiltered[:, 0], 
                                                        window_unfiltered[:, 1], 
                                                        window_unfiltered[:, 2], 
                                                        method = area_method,
                                                        filter = apply_filter,
                                                        best_combination = best_combination
                                                        )
        # calculate tapse
        (tapse_sep, 
            tapse_fw, 
            tapse) = tapse_calculation(window[:, 1], 
                                    window[:, 0], 
                                    window_unfiltered[:, 1], 
                                    window_unfiltered[:, 0], 
                                    tapse_calc=tapse_calc,
                                    direction=direction,
                                    filter = apply_filter,
                                    best_combination = best_combination)

        index_container[i] = [
            tapse_fw * 1000, 
            tapse_sep * 1000, 
            rvfac, 
            diast_area * 1e4, 
            syst_area * 1e4,
            rvldfw * 1000, 
            rvldsep * 1000, 
            rvlsfw * 1000, 
            rvlssep * 1000, 
            tadd * 1000,
            tasd * 1000, 
            rvldmid * 1000, 
            rvlsmid * 1000, 
            rvlsffw, 
            rvlsfglobal,
            rvlsfsep, 
            rvlsfmid
        ]

    if reduction == 'mean' and not best_combination:
         return index_container.mean(axis=0)
    elif reduction == 'max' and not best_combination:
        return index_container.max(axis=0)
    elif reduction == 'min' and not best_combination:
        return index_container.min(axis=0)
    elif best_combination:
        result = []
        for i in range(index_container.shape[1]):
            if i in [5] and best_rvlsffw:
                result.append(index_container[:, i].max())
            elif i in [0, 8]:
                result.append(index_container[:, i].mean())
            elif i in [1, 6, 10]:
                result.append(index_container[:, i].max())
            else:
                result.append(index_container[:, i].min())
        return np.asarray(result)



def main():
    parser = argparse.ArgumentParser(description="Predict RV indices from HDF5 files using a U-Net model.")

    parser.add_argument('--h5_dir', type=str, required=True, help='Directory containing the HDF5 files')
    parser.add_argument('--excel_path', type=str, required=True, help='Path to Excel file with patient metadata')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--depth', type=int, default=6, help='U-Net depth')
    parser.add_argument('--filters', type=int, default=12, help='Number of filters to start')
    parser.add_argument('--residuals', type=int, default=2, help='Number of residual units')
    parser.add_argument('--filter', action='store_true', help='Apply Kalman filter to coordinates')
    parser.add_argument('--tapse', type=str, choices=['distance', 'projection'], default='distance', help='TAPSE calculation method')
    parser.add_argument('--reduction', type=str, choices=['mean', 'max', 'min'], default='max', help='Reduction method for multiple beats')
    parser.add_argument('--images_path', type=str, default=None, help='Path to save images with keypoints (optional)')
    parser.add_argument('--save_images', action='store_true', help='Save images with keypoints')
    parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for center of mass detection') # default is 0.8, but must be adjusted based on the threshold used during training
    parser.add_argument('--two_dimensional', action='store_true', help='whether the images are 2d or 3d derived')
    parser.add_argument('--count_beats', action='store_true', help='Print number of detected heartbeats')
    parser.add_argument('--area_method', type=str, choices=['triangle', 'spline'], default='triangle', help='how to calculate the area inside the triangle')
    parser.add_argument('--best_combination', action='store_true', help='Use best combination of parameters found (overrides other parameters)')
    parser.add_argument('--no_sudden_movements', action='store_true', help='Flag to avoid sudden movements in keypoints (not implemented yet)')
    parser.add_argument('--threshold_sudden', type=int, default=20, help='Threshold for sudden movement detection')
    parser.add_argument('--best_rvlsffw', action='store_true', help='if the overestimate rvldfw, so that rvlsffw is more similar to the manual one. ' \
    'Then rvlsffw has to be recalculated as (rvldfw - rvlsfw)/rvldfw*100 from the excel results in the statistical analysis script. Use it in combination with --best_combination.')
    args = parser.parse_args()

    columns = [
        "tapsefw", 
        "tapsesep", 
        "rvfac", 
        "rvad", 
        "rvas",
        "rvldfw", 
        "rvldsep", 
        "rvlsfw", 
        "rvlssep", 
        "tadd",
        "tasd", 
        "rvldmid", 
        "rvlsmid", 
        "rvlsffw", 
        "rvlsfglobal",
        "rvlsfsep", 
        "rvlsfmid"
    ]

    df = pd.read_excel(args.excel_path)

    for col in columns:
        if col not in df.columns:
            df[col] = None

    if "path" not in df.columns:
        raise ValueError("Excel file must contain a 'path' column.")

    paths = df["path"].dropna().tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(depth=args.depth, start_filts=args.filters, num_residuals=args.residuals).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
    model.eval()

    for path in paths:
        test_path = os.path.join(args.h5_dir, str(path) + ".h5")
        print(f"Processing: {test_path}")

        try:
            indexes = predict_indices(model, 
                                        test_path, 
                                        apply_filter=args.filter, 
                                        device=device, 
                                        tapse_calc=args.tapse, 
                                        reduction=args.reduction, 
                                        patient = path, 
                                        save_images=args.save_images, 
                                        images_path=args.images_path, 
                                        threshold=args.threshold, 
                                        two_dimensional=args.two_dimensional, 
                                        count_beats=args.count_beats,
                                        area_method = args.area_method,
                                        best_combination = args.best_combination,
                                        no_sudden_movements = args.no_sudden_movements,
                                        threshold_sudden = args.threshold_sudden,
                                        best_rvlsffw = args.best_rvlsffw
                                      ) # function that predicts indexes froom the h5 file
            
            row_idx = df.index[df["path"] == path].tolist()
            if not row_idx:
                print(f"Path {path} not found in DataFrame.")
                continue
            row = row_idx[0]
            for i, col in enumerate(columns):
                df.at[row, col] = indexes[i]
        except Exception as e:
            print(f"Error processing {path}: {e}")

    df.to_excel(args.excel_path, index=False)
    print("Excel file updated.")


if __name__ == "__main__":
    main()