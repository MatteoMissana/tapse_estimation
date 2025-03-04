import os
import numpy as np
import torch
import torchvision.transforms.functional as F

def resize_or_crop_image(img, keypoints, target_size=(256, 256)):
    """
    Resizes or crops the input image to match the target size.
    If the image is larger, it is cropped centrally.
    If the image is smaller, it is padded symmetrically.
    """
    h, w = img.shape[-2:]
    #print(img.shape)
    
    # If image is larger, crop it centrally
    if h > target_size[0] and w > target_size[1]:
        crop_h = max(0, h - target_size[0]) // 2
        additional_h = abs(h - target_size[0]) % 2
        crop_w = max(0, w - target_size[1]) // 2
        additional_w = abs(w - target_size[1]) % 2
        img = img[crop_h:h - crop_h - additional_h, crop_w:w - crop_w - additional_w]
        keypoints[:, 0] -= crop_w
        keypoints[:, 1] -= crop_h

    elif h > target_size[0] and w < target_size[1]:
        crop_h = max(0, h - target_size[0]) // 2
        additional_h = abs(h - target_size[0]) % 2
        pad_w = (target_size[1] - w) // 2
        keypoints[:, 1] -= crop_h
        keypoints[:, 0] += pad_w
        
        img = img[crop_h:h - crop_h - additional_h]
        img = F.pad(img, (pad_w, 0, pad_w + (target_size[1] - w) % 2, 0), fill=0)
    
    elif h < target_size[0] and w > target_size[1]:
        pad_h = (target_size[0] - h) // 2
        crop_w = max(0, w - target_size[1]) // 2
        additional_w = abs(w - target_size[1]) % 2
        keypoints[:, 0] -= crop_w
        keypoints[:, 1] += pad_h

        img = img[:, crop_w:w - crop_w - additional_w]
        img = F.pad(img, (0, pad_h, 0, pad_h + (target_size[0] - h) % 2), fill=0)
    
    # If image is smaller, pad it
    else:
        pad_h = (target_size[0] - h) // 2
        pad_w = (target_size[1] - w) // 2
        keypoints[:, 0] += pad_w
        keypoints[:, 1] += pad_h
        
        img = F.pad(img, (pad_w, pad_h, pad_w + (target_size[1] - w) % 2, pad_h + (target_size[0] - h) % 2), fill=0)
    
    print(img.shape, keypoints.shape)
    return img, keypoints

def create_numpy_dataset(folder=r'D:\mmissana\data\best_slices', save_path=r'D:\mmissana\data\dataset_separated_by_video_256'):
    keypoints_np = {}
    images_np = {}
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for patient in os.listdir(folder):
        if patient == 'readme.txt':
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
                    image_processed, keypoints_processed = resize_or_crop_image(
                        torch.tensor(images[i], dtype=torch.float32), 
                        torch.tensor(keypoints[i], dtype=torch.float32)
                    )
                    image_list.append(image_processed.numpy())
                    keypoint_list.append(keypoints_processed.numpy())
                
                keypoints_np[patient] = np.array(keypoint_list)
                images_np[patient] = np.array(image_list)
                
                patient_save_path = os.path.join(save_path, f"{patient}.npz")
                np.savez_compressed(patient_save_path, images=images_np[patient], keypoints=keypoints_np[patient])
                
    print("Dataset creation complete.")

if __name__ == "__main__":
    create_numpy_dataset()