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
def create_numpy_dataset(folder = r'D:\mmissana\data\best_slices', save_path = 'dataset.npz'):
    """
    Reads images and keypoints from files in the specified folder, applies padding to
    images, and saves the dataset as a .npz file containing both images and keypoints.
    """
    image_list = []  # List to hold padded images
    keypoint_list = []  # List to hold keypoints corresponding to images
    
    # Loop through patients in the folder
    for patient in os.listdir(folder):
        if patient != 'readme.txt':  # Skip readme.txt file
            # Loop through documents in each patient's directory
            for doc in os.listdir(os.path.join(folder, patient)):
                if doc == 'video_best_slice_annotations.npz':  # Look for the specific annotations file
                    file_path = os.path.join(folder, patient, doc)
                    file = np.load(file_path)  # Load the .npz file
                    
                    # Extract frames (images) and annotations (keypoints) from the file
                    images = file['frames']
                    images = images.transpose(2, 0, 1)  # Reorder dimensions from (H, W, frames) to (frames, H, W)
                    keypoints = file['annotations']
                    
                    # Loop through each frame/image
                    for i in range(len(images)):
                        # Convert image to a tensor, apply padding, then convert back to NumPy array
                        image_padded, keypoints_padded = pad_image(torch.tensor(images[i], dtype=torch.float32), torch.tensor(keypoints[i], dtype=torch.float32))
                        image_padded = image_padded.numpy()
                        keypoints_padded = keypoints_padded.numpy()
                        image_list.append(image_padded)  # Add the padded image to the image list
                        keypoint_list.append(keypoints_padded)  # Add the corresponding keypoint to the keypoint list
    
    # Convert image and keypoint lists to NumPy arrays
    keypoints_np = np.array(keypoint_list)
    images_np = np.array(image_list)
    
    # Print the shapes of the images and keypoints arrays to verify
    print(images_np.shape)
    print(keypoints_np.shape)

    # Save the dataset to the specified file path as a .npz file
    np.savez(save_path, images=images_np, keypoints=keypoints_np)


# Main function that will execute when the script is run
if __name__ == "__main__":
    # Call the create_numpy_dataset function and save the dataset to the specified path
    create_numpy_dataset(save_path='data/dataset/dataset.npz')
