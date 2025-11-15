import os
import h5py
import torch
import numpy as np
import pandas as pd
import argparse

from dataloader.preprocessing import preprocess_images, apply_lut, resize_or_crop_image_np_nokeypoints
from models.models import Unet
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from postprocessing.indices_calculation import tric_apex_distance_calculation, tric_apex_distance_calculation_best, tapse_calculation, tapse_calculation_best, find_parallel_direction, find_es, find_ed_es_from_distance, indices_calculation
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
                    apply_filter='none', 
                    device='cpu', 
                    reduction='max', 
                    patient=None,  
                    threshold=0.875, 
                    two_dimensional=True, 
                    area_method = 'triangle',
                    best_combination = False,
                    threshold_sudden = 4, #2 mm
                    avg_all= False,
                    ):
    """Compute indices from a cardiac HDF5 sequence using a trained tracking model."""

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
    beat_start = [np.argmin(np.abs(images_times - ecg_times[r])) for r in r_peaks]

    # Inference loop
    coordinates_array = np.zeros((len(images), 3, 2))
    for i, im in enumerate(images):
        im = preprocess_images(np.expand_dims(im, axis=0), model_type='U-Net', device=device)
        output = model(im.float().unsqueeze(0).to(device))

        coords = [center_of_mass(output[0, c].detach(), thresh=threshold) for c in range(3)]
        for j in range(3):
            coordinates_array[i, j] = coords[j]

    if best_combination:
        #set threshold sudden to the best
        threshold_sudden = 4

    # since some indexes are better calculated from the kalman filtering, others from the avg., 
    # others from both, and others from none, I need to create the arrays to calculate everything
    # Actually I think it's easier to calculate each of the 4 arrays each time and use the one I want based on the type of filtering. 
    # (since also each index needs its own filtering for best performance)
    # It's more ordered and at the end of the day I will be using --best_combination argument most of the time
    coordinates_array_avg = coordinates_array.copy()
    both_filters_array = coordinates_array.copy()
    filtered_array = coordinates_array.copy()

    #only avg to the avg one
    for j in range(3):
        for i in range(1, len(coordinates_array_avg)-1):
            if np.linalg.norm(coordinates_array_avg[i, j] - coordinates_array_avg[i-1, j]) > threshold_sudden and np.linalg.norm(coordinates_array_avg[i, j] - coordinates_array_avg[i+1, j]) > threshold_sudden: 
                coordinates_array_avg[i, j] = (coordinates_array_avg[i-1, j] + coordinates_array_avg[i+1, j]) / 2


    # only Kalman to the filtered one
    kfs = [KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=0.1, y_std_meas=0.1) for _ in range(3)]
    for j in range(3):
        kfs[j].x = np.matrix([coordinates_array[0, j, 0], coordinates_array[0, j, 1], 0, 0]).T

    for i, coords in enumerate(coordinates_array):
        for j in range(3):
            kfs[j].predict()
            filtered_array[i, j] = kfs[j].update(np.matrix(coords[j]).T).A1

    # both to the bpth filters one
    for j in range(3):
        for i in range(1, len(both_filters_array)-1):
            if np.linalg.norm(both_filters_array[i, j] - both_filters_array[i-1, j]) > threshold_sudden and np.linalg.norm(both_filters_array[i, j] - both_filters_array[i+1, j]) > threshold_sudden: 
                both_filters_array[i, j] = (both_filters_array[i-1, j] + both_filters_array[i+1, j]) / 2
    kfs = [KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=0.1, y_std_meas=0.1) for _ in range(3)]
    for j in range(3):
        kfs[j].x = np.matrix([both_filters_array[0, j, 0], both_filters_array[0, j, 1], 0, 0]).T

    for i, coords in enumerate(both_filters_array):
        for j in range(3):
            kfs[j].predict()
            both_filters_array[i, j] = kfs[j].update(np.matrix(coords[j]).T).A1

    # so at the end of thsis block I have 4 arrays: 
    # coordinates_array = no filtering
    # filtered_array=Kalman filter
    # both_filters_array = both
    # coordinates_array_avg= only avg

    # Rescale to physical units
    filtered_array[:, :, 0] *= pixelsize[0]
    filtered_array[:, :, 1] *= pixelsize[1]

    both_filters_array[:, :, 0] *= pixelsize[0]
    both_filters_array[:, :, 1] *= pixelsize[1]

    coordinates_array_avg[:, :, 0] *= pixelsize[0]
    coordinates_array_avg[:, :, 1] *= pixelsize[1]

    coordinates_array[:, :, 0] *= pixelsize[0]
    coordinates_array[:, :, 1] *= pixelsize[1]

    #find the direction parallel to the fw movement
    direction = find_parallel_direction(coordinates_array[:, 0]) 
    direction /= np.linalg.norm(direction)

    if len(coordinates_array) - beat_start[-1] < 20:
        index_container = np.zeros((len(beat_start) - 1, 17))
    else:
        index_container = np.zeros((len(beat_start), 17))

    count = 0

    for i, frame_num in enumerate(beat_start):
        # I need to select a window from the calculated arrays (window = 1 heartbeat)
        if i == len(beat_start)-1: # if it's the last heart-beat
            if len(coordinates_array) - frame_num < 20: # if the last r peak is too close to the end, i dont consider the last "beat"
                print('jump heartbeat')
                continue
            else: # take the window from last r-peak to the end otherwise
                window_kalman = filtered_array[beat_start[i]:] 
                window_unfiltered = coordinates_array[beat_start[i]:]
                window_both = both_filters_array[beat_start[i]:]
                window_avg = coordinates_array_avg[beat_start[i]:]
        else: # take the window from the r-peak to the next one if it's not the las r-peak
            window_kalman = filtered_array[beat_start[i]:beat_start[i + 1]] 
            window_unfiltered = coordinates_array[beat_start[i]:beat_start[i + 1]]
            window_both = both_filters_array[beat_start[i]:beat_start[i + 1]]
            window_avg = coordinates_array_avg[beat_start[i]:beat_start[i + 1]]

        # find the es frame based on the maximum and minimum midpoint to apex distance
        es_pixel = find_es(window_both, direction=direction)
        ed_pixel = 0 # the end-diastole frame is frame 0 of the window since the window starts with the frame closest to the R-peak
        
        # fold = os.path.join(r"C:\Users\User\Desktop\test_es_ed_direction", patient.split('\\' )[0])
        # for j in range(len(window_both)):
        #     if j == 0:
        #         save_image(images[count+j], save_folder=fold, cmap='Blues') #ed_pixel
        #     elif j== es_pixel:
        #         save_image(images[count+j], save_folder=fold, cmap='Greens')#es pixel
        #     else:
        #         save_image(images[count+j], save_folder= fold, cmap='gray')
        # count = count + len(window_both)

        if apply_filter == 'none':
            window = window_unfiltered
        elif apply_filter == 'kalman':
            window = window_kalman
        elif apply_filter == 'both':
            window = window_both
        else: # apply_filter == 'avg'
            window = window_avg            

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
        rvlsfglobal,
        tapse_fw,
        tapse_sep) = indices_calculation(
            window[ed_pixel], 
            window[es_pixel],
            method = area_method,
            )
    
        # if not best_combination then the array has 17 values (all the indexes)
        if not best_combination:
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
                rvlsfmid,
            ]
        # otherwise the array also has 2 more values, that are only used in statystical_analysis.py to recalculate some indexes 
        # (necessary bbecause some indexes are recalculated with some overestimation of other indexes, so I want to extract both the index 
        # and its overestimation and then use the overestimation to recalcualte the second index)
        else:
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
                rvlsfmid,
                rvldfw_calc * 1000, # this is only used to calculate rv strain (fw) in statistical_analysis.py
                rvlsfw_calc * 1000, # this is only used to calculate rv strain (fw) in statistical_analysis.py
            ]
    



           
    if best_combination: # also extract 2 more indexes that are some intentional overestimations of rvldfw and rvlsfw. these are used later
        #(in statistical_analysis.py) to calculate rvlsffw and lower the bias
        # Compute metrics for each beat
        if not avg_all:
            # pick the maximum, minimum or mean value based on the index best performance
            result = []
            for i in range(index_container.shape[1]):
                if i in [0, 8]:
                    result.append(index_container[:, i].mean())
                elif i in [1, 6, 10, 14, 16, 17]:
                    result.append(index_container[:, i].max())
                else:
                    result.append(index_container[:, i].min())
            return np.asarray(result)
        else:
            return index_container.mean(axis=0)
     
    else: # not best_combination, manual selection of the method to use
        # pick max mean or min value across heartbeats based on specifications
        if reduction == 'mean' and not best_combination:
            return index_container.mean(axis=0)
        elif reduction == 'max' and not best_combination:
            return index_container.max(axis=0)
        elif reduction == 'min' and not best_combination:
            return index_container.min(axis=0)



def main():
    parser = argparse.ArgumentParser(description="Predict RV indices from HDF5 files using a U-Net model.")

    parser.add_argument('--h5_dir', type=str, required=True, help='Directory containing the HDF5 files')
    parser.add_argument('--excel_path', type=str, required=True, help='Path to Excel file where to save the data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--depth', type=int, default=6, help='U-Net depth')
    parser.add_argument('--filters', type=int, default=12, help='Number of filters to start')
    parser.add_argument('--residuals', type=int, default=2, help='Number of residual units')
    parser.add_argument('--filter', type = str, choices=['kalman','avg','none', 'both'], help='type of filter to apply')
    parser.add_argument('--reduction', type=str, choices=['mean', 'max', 'min'], default='max', help='Reduction method for multiple beats')
    parser.add_argument('--images_path', type=str, default=None, help='Path to save images with keypoints (optional)')
    parser.add_argument('--save_images', action='store_true', help='Save images with keypoints')
    parser.add_argument('--threshold', type=float, default=0.875, 
                        help='Threshold for center of mass detection') # default is 0.875, but must be adjusted based on the threshold used during training
    parser.add_argument('--two_dimensional', action='store_true', help='whether the images are 2d or 3d derived')
    parser.add_argument('--count_beats', action='store_true', help='Print number of detected heartbeats')
    parser.add_argument('--area_method', type=str, choices=['triangle', 'spline'], default='triangle', help='how to calculate the area inside ' \
    'the triangle')
    parser.add_argument('--best_combination', action='store_true', help='Use best combination of parameters found (overrides other parameters)')
    parser.add_argument('--threshold_avg', type=int, default=4, help='Threshold for avg filter application')
    parser.add_argument('--avg_all', action='store_true', help='if called together with --best combination, ' \
    'it uses an averaging reduction method instead of "cherry picking" the best method for each index. Ideally,' \
    ' with a perfect model, this would be the best way to proceed')
    args = parser.parse_args()

    if args.best_combination:
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
            "rvlsfmid",
            "rvldfw (only for rv strain calculation)",
            "rvlsfw (only for rv strain calculation)",
        ]
    else:
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
            "rvlsfmid",
        ]

    df = pd.read_excel(args.excel_path)

    for col in columns:
        if col not in df.columns:
            df[col] = None

    if "path" not in df.columns:
        raise ValueError("Excel file must contain a 'path' column.")

    paths = df["path"].dropna().tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #best model from experiments, change this part (and the landmark calculations from feature maps probably) if you want to use a different model
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
                                        reduction=args.reduction, 
                                        patient = path,
                                        threshold=args.threshold, 
                                        two_dimensional=args.two_dimensional, 
                                        area_method = args.area_method,
                                        best_combination = args.best_combination,
                                        threshold_sudden = args.threshold_avg,
                                        avg_all = args.avg_all,
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