import h5py

# Path to one of your new files
file_path = 'data/RV_PATIENTS/RV_patients_predicted/_11010/P42A0G2A.h5'

# Open the file in read mode
with h5py.File(file_path, 'r') as h5_file:
    # Print all top-level datasets/groups
    print("Top-level keys in the HDF5 file:")
    for key in h5_file.keys():
        print(f" - {key}")

    # Check if 'keypoints' exists
    if 'keypoints' in h5_file:
        keypoints = h5_file['keypoints'][()]
        print("\nKeypoints shape:", keypoints.shape)
        print("Example keypoints (first frame):\n", keypoints[0])
    else:
        print("No 'keypoints' dataset found.")
