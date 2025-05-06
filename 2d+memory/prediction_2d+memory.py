# Import necessary libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py

# Import custom preprocessing, model, and postprocessing utilities
from dataloader.preprocessing import preprocess_images, apply_lut, resize_or_crop_image_np_nokeypoints
from dataloader.dataset_creation_memory import generate_image_with_gaussians
from models.tasken_unet_with_memory import UNet as Unet_memory
from models.tasken_unet import UNet
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from postprocessing.indices_calculation import tric_apex_distance_calculation
from postprocessing.kalman_filter import KalmanFilter
from postprocessing.cardiac_phase_detection import systole_diatole_detection
from postprocessing.pan_tompkins import pan_tompkins_detector
from utils.plot import visualize_image

# Path to test data and trained model
test_path = r'D:\mmissana\data/RV_PATIENTS/RV_patients_converted/_195219/P429Q4PS.h5'
model_checkpoint = r'D:\mmissana\runs\unet_augm7_gaussian_curve_training\best_model.pth'
model_checkpoint_memory = r'D:\mmissana\tapse_estimation/2d+memory/runs/unet_with_memory/best_model.pth'

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the trained U-Net model
model = UNet(num_classes=3, depth=6, start_filts=8).to(device)
model_2 = Unet_memory(num_classes=3, depth=6, start_filts=8).to(device)
model.load_state_dict(torch.load(model_checkpoint, map_location=device)['model_state_dict'])
model_2.load_state_dict(torch.load(model_checkpoint_memory, map_location=device)['model_state_dict'])

def predict_indices(model, test_path):
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

        # Compute sampling frequency of ECG
        fs = 1 / (ecg_times[1] - ecg_times[0])

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

    apex_distances = []
    filtered_distances = []
    diameters = []
    filtered_diameters = []

    # Initialize Kalman filter for smoothing distance measurements
    kalman_filter = KalmanFilter(process_variance=0.05, measurement_variance=5e-2, initial_estimate=0, initial_error=10)
    kalman_diameter = KalmanFilter(process_variance=0.05, measurement_variance=5e-2, initial_estimate=0, initial_error=10)

    # Detect R-peaks in ECG using Pan-Tompkins algorithm
    r_peaks = pan_tompkins_detector(ecg, fs)

    # Find corresponding image frame indices for each R-peak
    beat_start = []
    for r in r_peaks:
        time_diff = np.abs(images_times - ecg_times[r])
        closest_index = np.argmin(time_diff)
        beat_start.append(closest_index)

    flag_first = True
    for i, im in enumerate(images):
        # Prepare image for input to model
        im = np.expand_dims(im, axis=0)
        im = preprocess_images(im, model_type='U-Net', device=device)
        im = im.float().unsqueeze(0).to(device)
        

        if flag_first:
            # Run segmentation model
            output = model(im)
        else:
            gaussian_map = generate_image_with_gaussians(256, [[coordinates_1.tolist(), coordinates_2.tolist(), coordinates_3.tolist()]], std=10.0).to(device)
            gaussian_map = gaussian_map.unsqueeze(0).unsqueeze(0)
            im = torch.cat((im, gaussian_map), dim=2)
            output = model_2(im)

        # Extract coordinates of segmented structures via center of mass
        coordinates_1 = center_of_mass(output[0, 0].detach())
        coordinates_2 = center_of_mass(output[0, 1].detach())
        coordinates_3 = center_of_mass(output[0, 2].detach())

        # visualize_image(im[0, 0,0].cpu().numpy(), points=[tuple(coordinates_1.tolist()), tuple(coordinates_2.tolist()), tuple(coordinates_3.tolist())])

        # Calculate distance between tricuspid valve and apex
        apex_dist, diameter = tric_apex_distance_calculation(coordinates_1, coordinates_2, coordinates_3)

        if flag_first:
            kalman_filter.initial_estimate = apex_dist
            kalman_diameter.initial_estimate = diameter
            flag_first = False

        # Apply Kalman filter for temporal smoothing
        filtered_distance = kalman_filter.update(apex_dist)
        filtered_diameter = kalman_diameter.update(diameter)

        apex_distances.append(apex_dist)
        filtered_distances.append(filtered_distance)
        diameters.append(diameter)
        filtered_diameters.append(filtered_diameter)
        filtered_array = np.array(filtered_distances)
        filtered_diameters_array = np.array(filtered_diameters)

    # Optional plot for visual inspection (currently commented out)
    # plt.figure(figsize=(12, 6))
    # plt.plot(apex_distances, label="Raw Distance", linestyle='--', marker='o')
    # plt.plot(filtered_distances, label="Filtered Distance", linestyle='-', marker='x')
    # plt.xlabel("Frame Index")
    # plt.ylabel("Distance")
    # plt.title("Tricuspid-Apex Distance Over Time")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    rvlsf = []

    # Compute RVLSF for each heartbeat segment
    for i, start in enumerate(beat_start):
        if start != 0 and start != len(images) and i < len(beat_start) - 1:
            window = filtered_array[beat_start[i]:beat_start[i+1]]

            diast_distance = window.max()
            syst_distance = window.min()

            # RVLSF is the relative shortening between diastole and systole
            rvlsf.append(((diast_distance - syst_distance) / diast_distance) * 100)

    rvlsf = np.array(rvlsf)
    return rvlsf.mean()  # Return mean RVLSF across all cycles

# Entry point to execute the function and print the result
if __name__ == "__main__":
    rvlsf = predict_indices(model, test_path)
    print(f"RVLSF: {rvlsf:.2f}%")