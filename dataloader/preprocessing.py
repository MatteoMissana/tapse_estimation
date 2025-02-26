import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# Function to pad an image to the target size
def pad_image(img, keypoints, target_size=(360, 360)):
    """
    Pads the input image to match the target size. It adds padding symmetrically
    to both height and width. If the difference in size is odd, the padding is adjusted.
    """
    h, w = img.shape[-2:]  # Get the height (h) and width (w) of the image
    pad_h = (target_size[0] - h) // 2  # Calculate vertical padding
    pad_w = (target_size[1] - w) // 2  # Calculate horizontal padding

    keypoints[:, 0] += pad_w
    keypoints[:, 1] += pad_h

    # Check if both height and width require odd padding
    if (target_size[0] - h) % 2 != 0 and (target_size[1] - w) % 2 != 0:
        # Add 1 extra pixel padding for both height and width if the remaining difference is odd
        img = F.pad(img, (pad_w, pad_h, pad_w+1, pad_h+1), fill=0)
    
    # If only height requires odd padding
    elif (target_size[0] - h) % 2 != 0:
        img = F.pad(img, (pad_w, pad_h, pad_w, pad_h+1), fill=0)
      
    # If only width requires odd padding
    elif (target_size[1] - w) % 2 != 0:
        img = F.pad(img, (pad_w, pad_h, pad_w+1, pad_h), fill=0)
    
    # If no odd padding needed, just apply normal symmetric padding
    else:
        img = F.pad(img, (pad_w, pad_h, pad_w, pad_h), fill=0)

    return img, keypoints


# Function to create a dataset from images and annotations
# Function to create a dataset from images and annotations
def create_numpy_dataset(folder=r'D:\mmissana\data\best_slices', save_path=r'D:\mmissana\data\dataset_separated_by_video'):
    keypoints_np = {}  
    images_np = {}  

    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Ensure the save path exists

    for patient in os.listdir(folder):
        if patient == 'readme.txt':  # Skip any unwanted files
            continue

        image_list = []  
        keypoint_list = []  
        
        patient_folder = os.path.join(folder, patient)

        for doc in os.listdir(patient_folder):
            if doc == 'video_best_slice_annotations.npz':
                file_path = os.path.join(patient_folder, doc)
                file = np.load(file_path)

                images = file['frames'].transpose(2, 0, 1)  # (H, W, frames) → (frames, H, W)
                keypoints = file['annotations']

                print(patient, 'number of images:', len(images))

                for i in range(len(images)):
                    image_padded, keypoints_padded = pad_image(
                        torch.tensor(images[i], dtype=torch.float32), 
                        torch.tensor(keypoints[i], dtype=torch.float32)
                    )
                    image_list.append(image_padded.numpy())
                    keypoint_list.append(keypoints_padded.numpy())

        
                keypoints_np[patient] = np.array(keypoint_list)
                images_np[patient] = np.array(image_list)

                # Save each patient's data separately
                patient_save_path = os.path.join(save_path, f"{patient}.npz")
                np.savez_compressed(patient_save_path, images=images_np[patient], keypoints=keypoints_np[patient])
                # print(f"Saved: {patient_save_path}")

    print("Dataset creation complete.")


# Main function that will execute when the script is run
if __name__ == "__main__":
    # Call the create_numpy_dataset function and save the dataset to the specified path
    create_numpy_dataset()
