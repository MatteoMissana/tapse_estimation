import h5py
from utils.plot import VolumeViewer
import numpy as np
import os
import json
from utils.extract_slices import extract_from_hdf5

'''code that i used to extract the 2d slices from the 3d volume'''

folder = r"D:\mmissana\data\4DRVQ_Jinyang\voxels"
save_folder = r"D:\mmissana\data\processed_imgs_2"
checkpoint_file = r"D:\mmissana\data\checkpoint_2.json"

def load_checkpoint():
    """Load checkpoint file if it exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoint_data):
    """Save checkpoint data."""
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=4)

degrees = np.linspace(np.pi, 2*np.pi, 18)
checkpoint = load_checkpoint()  # Load progress

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    print(f"Processing {file}...")

    # Skip already processed files
    if file in checkpoint and checkpoint[file] == "complete":
        print(f"Skipping {file}, already processed.")
        continue

    # Manually enter tric and apex values
    tric_values = input(f"Enter 3 values for tric (space-separated) for {file}: ")
    apex_values = input(f"Enter 3 values for apex (space-separated) for {file}: ")

    try:
        tric = np.array([float(x) for x in tric_values.split()])
        apex = np.array([float(x) for x in apex_values.split()])
        
        if tric.shape != (3,) or apex.shape != (3,):
            raise ValueError("Tric and Apex must each have exactly 3 values.")

    except ValueError as e:
        print(f"Error in input values: {e}")
        continue
    
    # Extract slices from the volume
    save_subfolder = os.path.join(save_folder, file.replace(".h5", ""))
    os.makedirs(save_subfolder, exist_ok=True)
    extract_from_hdf5(file_path, save_subfolder, degrees, tric=tric, apex=apex)

    # Save the checkpoint
    checkpoint[file] = "complete"
    save_checkpoint(checkpoint)
    print(f"File {file} processed successfully!")

