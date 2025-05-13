# Import necessary libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py

# Import custom preprocessing, model, and postprocessing utilities
from dataloader.preprocessing import preprocess_images, apply_lut, resize_or_crop_image_np_nokeypoints
from models.tasken_unet import UNet
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from postprocessing.indices_calculation import tric_apex_distance_calculation, tapse_calculation, find_parallel_direction
from postprocessing.kalman_filter import KalmanFilter
from postprocessing.cardiac_phase_detection import systole_diatole_detection
from postprocessing.pan_tompkins import pan_tompkins_detector
from utils.plot import visualize_image


def print_attrs(name, obj):
    print(f"Name: {name}")
    for key, val in obj.attrs.items():
        print(f"  Attr: {key} => {val}")



# Path to test data and trained model
test_path = r'D:\mmissana\data/RV_PATIENTS/RV_patients_annotated/_794029/P429K79A.h5'
model_checkpoint = r'D:\mmissana\runs\unet_augm7_gaussian_curve_training\best_model.pth'

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the trained U-Net model
model = UNet(num_classes=3, depth=6, start_filts=8, in_channels=3).to(device)
model.load_state_dict(torch.load(model_checkpoint, map_location=device)['model_state_dict'])

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
        # f.visititems(print_attrs)
        pixelsize = f['tissue']['pixelsize'][()]

        # Compute sampling frequency of ECG
        fs = 1 / (ecg_times[1] - ecg_times[0])

        #compute sampling frequency of images
        fs_images = 1 / (images_times[1] - images_times[0])
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

    apex_distances = []
    filtered_distances = []
    diameters = []
    filtered_diameters = []

    # Detect R-peaks in ECG using Pan-Tompkins algorithm
    r_peaks = pan_tompkins_detector(ecg, fs, plot=False)
    print(f"R-peaks detected at indices: {r_peaks}")

    # Find corresponding image frame indices for each R-peak
    beat_start = []
    for r in r_peaks:
        time_diff = np.abs(images_times - ecg_times[r])
        closest_index = np.argmin(time_diff)
        beat_start.append(closest_index)

    #TODO: identify whole heartbeats
    # if beat_start[0] == 0:
    #     beat_start.pop(0) #remove if the beat start is the first value (could be a local maximum)
    # if beat_start[-1] == len(images):
    #     beat_start.pop(-1) #remove if the beat start is the last value (could be a local maximum)
    # beat_start.pop(-1) #remove another r peak (so that the last heartbeat will be complete since I take all the values after each r peak) 

    flag_first = True
    coordinates_array = np.zeros((len(images), 3, 2))

    for i, im in enumerate(images):
        # Prepare image for input to model
        im = np.expand_dims(im, axis=0)
        im = preprocess_images(im, model_type='U-Net', device=device)
        im = im.float().unsqueeze(0).to(device)
        im = im.repeat(1, 1, 3, 1, 1)

        # Run deteciton model
        output = model(im)

        # Extract coordinates of segmented structures via center of mass method
        coordinates_1 = center_of_mass(output[0, 0].detach())
        coordinates_2 = center_of_mass(output[0, 1].detach())
        coordinates_3 = center_of_mass(output[0, 2].detach())

        # visualize_image(im[0, 0, 0].cpu().numpy(), [coordinates_1, coordinates_2, coordinates_3])

        coordinates_array[i, 0] = coordinates_1
        coordinates_array[i, 1] = coordinates_2
        coordinates_array[i, 2] = coordinates_3

    # plt.figure(figsize=(6, 6))
    # plt.scatter(coordinates_array[:,0,0], coordinates_array[:,0,1], color='blue', label='free_wall')
    # plt.scatter(coordinates_array[:,1,0], coordinates_array[:,1,1], color='red', label='septum')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Punti 2D')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.legend()
    # plt.show()

    #filter the coordinates to remove noise
    kf1 = KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=.1, y_std_meas=.1)
    kf2 = KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=.1, y_std_meas=.1)
    kf3 = KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=.1, y_std_meas=.1)

    # print(dt)
    #initialize the position
    kf1.x = np.matrix([coordinates_array[0, 0, 0], coordinates_array[0, 0, 1], 0, 0]).T
    kf2.x = np.matrix([coordinates_array[0, 1, 0], coordinates_array[0, 1, 1], 0, 0]).T
    kf3.x = np.matrix([coordinates_array[0, 2, 0], coordinates_array[0, 2, 1], 0, 0]).T

    filtered_array = np.zeros_like(coordinates_array)
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

    

    for j in range(3):  # for each of the 3 points
        plt.plot(coordinates_array[:, j, 0], coordinates_array[:, j, 1], 'r', label=f'Original {j+1}' if j==0 else "")
        plt.plot(filtered_array[:, j, 0], filtered_array[:, j, 1], 'g', label=f'Filtered {j+1}' if j==0 else "")

    # plt.legend()
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Original vs Filtered Coordinates (2D)")
    # plt.grid(True)
    # plt.show()

    direction_free_wall = find_parallel_direction(coordinates_array[:, 0]) # direction parallel to thew movement of the free wall
    direction_septum = find_parallel_direction(coordinates_array[:, 1]) # direction parallel to thew movement of the septum

    average_direction = direction_free_wall + direction_septum
    average_direction /= np.linalg.norm(average_direction)

    rvlsf = []

    # Compute RVLSF for each heartbeat segment
    max_diameters = []
    min_diameters = []
    tapse_list = []
    for i in range(len(beat_start)-1):
        window = filtered_array[beat_start[i]:beat_start[i+1]]

        # Calculate distances and diameters
        apex_distance, diameter = tric_apex_distance_calculation(window[:,0], window[:,1], window[:,2])
        max_diameter = diameter.max()   
        min_diameter = diameter.min() 
        diast_distance = apex_distance.max()
        syst_distance = apex_distance.min()

        # calculate tapse
        tapse = tapse_calculation(coordinates_fw=window[:, 0], coordinates_septum=window[:, 1], direction = average_direction)

        # # Optional plot for visual inspection (currently commented out)
        # plt.figure(figsize=(12, 6))
        # plt.plot(apex_distance, label="Raw Distance", linestyle='--', marker='o')
        # plt.xlabel("Frame Index")
        # plt.ylabel("Distance")
        # plt.title("Tricuspid-Apex Distance Over Time")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # create arrays to average the measurement for all the heartbeats
        rvlsf.append(((diast_distance - syst_distance) / diast_distance) * 100)
        max_diameters.append(max_diameter)
        min_diameters.append(min_diameter)
        tapse_list.append(tapse)

    rvlsf = np.array(rvlsf)
    max_diameters_array = np.array(max_diameters)
    min_diameters_array = np.array(min_diameters)
    tapse = np.array(tapse_list)

    return rvlsf.mean(), min_diameters_array.mean(), max_diameters_array.mean(), tapse.mean() # Return mean RVLSF across all cycles

# Entry point to execute the function and print the result
if __name__ == "__main__":
    rvlsf, min_diam, max_diam, tapse= predict_indices(model, test_path)
    print(f"RVLSF: {rvlsf:.2f}%")
    print(f"Min Diameter: {min_diam*1000} mm")
    print(f"Max Diameter: {max_diam*1000} mm")
    print(f"TAPSE: {tapse*1000} mm")