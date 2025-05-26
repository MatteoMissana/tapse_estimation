# Import necessary libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import os

# Import custom preprocessing, model, and postprocessing utilities
from dataloader.preprocessing import preprocess_images, apply_lut, resize_or_crop_image_np_nokeypoints
from dataloader.dataset_creation_memory import generate_image_with_gaussians
from models.tasken_unet_with_memory import UNet as Unet_memory
from models.tasken_unet import UNet
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from postprocessing.indices_calculation import tric_apex_distance_calculation, tapse_calculation, find_parallel_direction
from postprocessing.kalman_filter import KalmanFilter
from postprocessing.cardiac_phase_detection import systole_diatole_detection
from postprocessing.pan_tompkins import pan_tompkins_detector
from utils.plot import visualize_image

def predict_indices(model, model_memory, test_path, filter=False):
    """
    Predicts the RVLSF index from a cardiac sequence using a trained segmentation model.
    
    Args:
        model: Trained PyTorch model for image segmentation.
        test_path: Path to the .h5 file containing the ultrasound images and ECG data.
    
    Returns:
        Mean RVLSF index over detected heartbeats.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image and ECG data from HDF5 file
    with h5py.File(test_path, 'r') as f:
        images = f['tissue']['data'][()]
        images_times = f['tissue']['times'][()]
        ecg = f['ecg']['ecg_data'][()]
        ecg_times = f['ecg']['ecg_times'][()]
        pixelsize = f['tissue']['pixelsize'][()]
        print(f"Pixelsize: {pixelsize}")

        # Compute sampling frequency of ECG
        fs = 1 / (ecg_times[1] - ecg_times[0])
        # Compute sampling frequency of images
        dt = images_times[1] - images_times[0]

        # Reorder image dimensions: (frames, height, width) → (height, width, frames)
        images = images.transpose(1, 0, 2)

        # Flip images vertically (correct orientation)
        images = images[:, ::-1, :]

        # Apply LUT and preprocess images
        images = apply_lut(images)
        images = images.transpose(2, 0, 1)  # Back to (frames, H, W)
        images = resize_or_crop_image_np_nokeypoints(images)

    # Normalize pixel values if necessary
    if images.max() > 1:
        images = images / 255.0


    # Detect R-peaks in ECG using Pan-Tompkins algorithm
    r_peaks = pan_tompkins_detector(ecg, fs, plot=False)
    print(f"R-peaks detected at indices: {r_peaks}")

    # Find corresponding image frame indices for each R-peak
    beat_start = []
    for r in r_peaks:
        time_diff = np.abs(images_times - ecg_times[r])
        closest_index = np.argmin(time_diff)
        beat_start.append(closest_index)

    print(f"Beat start indices: {beat_start}")


    coordinates_array = np.zeros((len(images), 3, 2))

    for i, im in enumerate(images):
        # Prepare image for input to model
        im = np.expand_dims(im, axis=0)
        im = preprocess_images(im, model_type='U-Net', device=device)
        im = im.float().unsqueeze(0).to(device)
        # im = im.repeat(1, 1, 3, 1, 1)

        if i != 0:
            model = model_memory

            gaussian_map = generate_image_with_gaussians(256, [coordinates_array[i].tolist()], std=10.0).to(device)
            gaussian_map = gaussian_map.unsqueeze(0).unsqueeze(0)

            im = torch.cat((im, gaussian_map), dim=2)

        # Run detection model
        output = model(im)

        # Extract coordinates of segmented structures via center of mass method
        coordinates_1 = center_of_mass(output[0, 0].detach())
        coordinates_2 = center_of_mass(output[0, 1].detach())
        coordinates_3 = center_of_mass(output[0, 2].detach())

        # if i % 10 == 0:
        #     visualize_image(im[0, 0, 0].cpu().numpy(), [coordinates_1, coordinates_2, coordinates_3])

        coordinates_array[i, 0] = coordinates_1
        coordinates_array[i, 1] = coordinates_2
        coordinates_array[i, 2] = coordinates_3

    filtered_array = coordinates_array.copy()
    if filter:
        # Filter the coordinates to remove noise
        kf1 = KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=.1, y_std_meas=.1)
        kf2 = KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=.1, y_std_meas=.1)
        kf3 = KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=.1, y_std_meas=.1)

        # print(dt)
        # Initialize the position
        kf1.x = np.matrix([coordinates_array[0, 0, 0], coordinates_array[0, 0, 1], 0, 0]).T
        kf2.x = np.matrix([coordinates_array[0, 1, 0], coordinates_array[0, 1, 1], 0, 0]).T
        kf3.x = np.matrix([coordinates_array[0, 2, 0], coordinates_array[0, 2, 1], 0, 0]).T

        # coordinates_array[10] = np.ones_like(coordinates_array[10])*300
        for i, predicted_coordinates in enumerate(coordinates_array):
            kf1.predict()
            kf2.predict()
            kf3.predict()

            filt1 = kf1.update(np.matrix(predicted_coordinates[0]).T)  # Correct with measurement
            filt2 = kf2.update(np.matrix(predicted_coordinates[1]).T)  # Correct with measurement
            filt3 = kf3.update(np.matrix(predicted_coordinates[2]).T)  # Correct with measurement

            filtered_array[i, 0] = filt1.A1  # or filt1.flatten()
            filtered_array[i, 1] = filt2.A1
            filtered_array[i, 2] = filt3.A1
            # print(f'predicted_coordinates {i}', predicted_coordinates)
            # print(f"filtered_array {i}", filtered_array[i])

    filtered_array[:,:,0] = filtered_array[:,:,0] * pixelsize[0]
    filtered_array[:,:,1] = filtered_array[:,:,1] * pixelsize[1]

    direction_free_wall = find_parallel_direction(coordinates_array[:, 0])  # direction parallel to the movement of the free wall
    direction_septum = find_parallel_direction(coordinates_array[:, 1])  # direction parallel to the movement of the septum

    # average_direction = direction_free_wall + direction_septum
    # average_direction /= np.linalg.norm(average_direction)

    # Compute RVLSF for each heartbeat segment
    index_container = np.zeros((len(beat_start)-1, 17))
    for i in range(len(beat_start)-1):
        window = filtered_array[beat_start[i]:beat_start[i+1]]

        # Calculate distances and diameters and areas
        rvfac, diast_area, syst_area, rvldfw, rvldsep, rvlsfw, rvlssep, rvldmid, rvlsmid, tadd, tasd, rvlsffw, rvlsfsep, rvlsfmid, rvlsfglobal = tric_apex_distance_calculation(window[:,0], window[:,1], window[:,2])

        # Calculate TAPSE
        tapse_sep, tapse_fw, tapse = tapse_calculation(coordinates_fw=window[:, 0], coordinates_septum=window[:, 1], direction_septum=direction_septum, direction_fw=direction_free_wall)

        index_container[i, 0] = tapse_fw*1000  # Convert to mm
        index_container[i, 1] = tapse_sep*1000  # Convert to mm
        index_container[i, 2] = rvfac
        index_container[i, 3] = diast_area*10000  # Convert to cm²
        index_container[i, 4] = syst_area*10000  # Convert to cm²
        index_container[i, 5] = rvldfw*1000  # Convert to mm
        index_container[i, 6] = rvldsep*1000  # Convert to mm
        index_container[i, 7] = rvlsfw*1000  # Convert to mm
        index_container[i, 8] = rvlssep*1000  # Convert to mm
        index_container[i, 9] = tadd*1000  # Convert to mm
        index_container[i, 10] = tasd*1000  # Convert to mm
        index_container[i, 11] = rvldmid*1000  # Convert to mm
        index_container[i, 12] = rvlsmid*1000  # Convert to mm
        index_container[i, 13] = rvlsffw
        index_container[i, 14] = rvlsfglobal
        index_container[i, 15] = rvlsfsep
        index_container[i, 16] = rvlsfmid

    return index_container.mean(axis = 0)  # Return mean RVLSF across all cycles


# Entry point to execute the function and print the result
# Entry point to execute the function and print the result
if __name__ == "__main__":
    h5_path = r'D:\mmissana\data\RV_PATIENTS\RV_patients_annotated'
    excel_path = r"D:\mmissana\data/RV_PATIENTS/Results_memory/UNet_memory.xlsx"

    model_checkpoint = r'D:\mmissana\tapse_estimation/2d/runs/Unet_1_channel/best_model.pth'
    model_checkpoint_memory = r'D:\mmissana\tapse_estimation/2d+memory/runs/Unet_memory_2_channels/best_model.pth'

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load the trained U-Net model
    model = UNet(num_classes=3, depth=6, start_filts=8, in_channels = 1).to(device)
    model_2 = Unet_memory(num_classes=3, depth=6, start_filts=8, in_channels = 2).to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)['model_state_dict'])
    model_2.load_state_dict(torch.load(model_checkpoint_memory, map_location=device)['model_state_dict'])


    columns = [
        "tapsefw", "tapsesep", "rvfac", "rvad", "rvas",
        "rvldfw", "rvldsep", "rvlsfw", "rvlssep", "tadd",
        "tasd", "rvldmid", "rvlsmid", "rvlsffw", "rvlsfglobal",
        "rvlsfsep", "rvlsfmid"
    ]

    # Load the DataFrame
    df = pd.read_excel(excel_path)

    # Make sure all columns are present (add them if missing)
    for col in columns:
        if col not in df.columns:
            df[col] = None  # initialize with null values

    # Extract paths
    if "path" in df.columns:
        paths = df["path"].dropna().tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(model_checkpoint, map_location=device)['model_state_dict'])

    for path in paths:
        test_path = os.path.join(h5_path, path + ".h5")
        print(f"Processing file: {test_path}")

        indexes = predict_indices(model, model_2, test_path, filter=True)

        # Find the row index corresponding to the current path
        row_idx = df.index[df["path"] == path].tolist()
        if not row_idx:
            print(f"Path {path} not found in DataFrame.")
            continue
        row = row_idx[0]

        # Write values into the correct row
        for i, col in enumerate(columns):
            df.at[row, col] = indexes[i]

    # Save the updated Excel file
    df.to_excel(excel_path, index=False)
    print("Excel file successfully updated.")