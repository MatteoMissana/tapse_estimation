import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py


from dataloader.preprocessing import preprocess_images, apply_lut, resize_or_crop_image_np_nokeypoints
from models.tasken_unet import UNet
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from postprocessing.indices_calculation import tric_apex_distance_calculation
from postprocessing.kalman_filter import KalmanFilter
from postprocessing.cardiac_phase_detection import systole_diatole_detection
from postprocessing.pan_tompkins import pan_tompkins_detector
from utils.plot import visualize_image

test_path = r'D:\mmissana\data/RV_PATIENTS/RV_patients_converted/_7101239/P42A69QU.h5'

model_checkpoint = r'D:\mmissana\runs\unet_augm7_gaussian_curve_training\best_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(num_classes=3, depth=6, start_filts=8).to(device)
model.load_state_dict(torch.load(model_checkpoint, map_location=device)['model_state_dict'])

def predict_indices(model, test_path):
    """
    Predicts the indices of the tricuspid valve using a trained model
    args:
        model: trained model
        test_path: path to the h5 file with images and ecg data
    returns:
        rvlsf: mean value of the RVLSF from the heartcycles
        tapse: same for tapse
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(test_path, 'r') as f:
        images = f['tissue']['data'][()]
        images_times = f['tissue']['times'][()]
        ecg = f['ecg']['ecg_data'][()]
        ecg_times = f['ecg']['ecg_times'][()]

        # Get the sampling frequency from the ECG times
        fs = 1 / (ecg_times[1] - ecg_times[0])

        # Rearrange axes so the frame format matches expected input (H, W, C)
        images = images.transpose(1, 0, 2)

        # Flip the frames vertically (mirror image over horizontal axis)
        images = images[:, ::-1, :]
        images = apply_lut(images)
        images = images.transpose(2,0,1)
        images = resize_or_crop_image_np_nokeypoints(images)

    if images.max() > 1:
        images = images / 255.0

    apex_distances = []
    filtered_distances = []

    kalman_filter = KalmanFilter(process_variance= .05, measurement_variance=5e-2, initial_estimate=0, initial_error=10)

    r_peaks = pan_tompkins_detector(ecg, fs)

    beat_start = []
    for r in r_peaks:
        time_diff = np.abs(images_times - ecg_times[r])
        closest_index = np.argmin(time_diff)
        beat_start.append(closest_index)

    flag_first = True
    for i, im in enumerate(images):
        im = np.expand_dims(im, axis=0)
        im = preprocess_images(im, model_type='U-Net', device=device)

        im = im.float()
        im = im.unsqueeze(0).to(device)

        output = model(im)

        coordinates_1 = center_of_mass(output[0, 0].detach())
        coordinates_2 = center_of_mass(output[0, 1].detach())
        coordinates_3 = center_of_mass(output[0, 2].detach())

        apex_dist = tric_apex_distance_calculation(coordinates_1, coordinates_2, coordinates_3)
        if flag_first:
            flag_first = False

        filtered_value = kalman_filter.update(apex_dist)

        apex_distances.append(apex_dist)
        filtered_distances.append(filtered_value)
        filtered_array = np.array(filtered_distances)
        

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

    for i, start in enumerate(beat_start):
        if start != 0 and start != len(images) and i < len(beat_start) - 1:
            window = filtered_array[beat_start[i]:beat_start[i+1]] 

            diast_distance = window.max()
            syst_distance = window.min()
            rvlsf.append(((diast_distance - syst_distance) / diast_distance) * 100)

    rvlsf = np.array(rvlsf)
    return rvlsf.mean()

if __name__ == "__main__":
    rvlsf = predict_indices(model, test_path)
    print(f"RVLSF: {rvlsf:.2f}%")