import torch
from torch.utils.data import Dataset
import numpy as np
import os
from dataloader.preprocessing import preprocess_images
from torchvision import transforms as T
from torchvision.transforms import v2 as T
from augmentations.augm_3d import apply_transform
from utils.plot import visualize_image


class KeypointDataset(Dataset):
    def __init__(self, numpy_dataset, transform=None, filter=False, preprocessing= False, device='cpu', model_type = 'U-Net'):
        """
        Args:
            images (list of np.array): List of grayscale images as numpy arrays.
            keypoints (list of lists): List of keypoint coordinates [[x1, y1, x2, y2], ...].
            transform (callable, optional): Image transformations.
            filter (bool, optional): If True, removes images with keypoints (0,0,0,0).
        """

        file = np.load(numpy_dataset)
        images = file['images']
        keypoints = file['keypoints']

        if filter:
            # Finding unannotated keypoints
            unannotated_indices = np.where(np.all(keypoints == 0, axis=1))[0]

            # Finding out-of-bounds keypoints
            out_of_bounds_indices = np.where(
                (keypoints[:, 0] < 0) | (keypoints[:, 0] > 256) |
                (keypoints[:, 1] < 0) | (keypoints[:, 1] > 256)
            )[0]

            # Combining both cases
            invalid_indices = np.unique(np.concatenate((unannotated_indices, out_of_bounds_indices)))

            images = np.delete(images, invalid_indices, axis=0)
            keypoints = np.delete(keypoints, invalid_indices, axis=0)

        self.images = images
        self.keypoints = keypoints
        self.transform = transform
        self.preprocessing = preprocessing
        self.device = device
        self.model_type = model_type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.max() > 1:
            img = img.astype(np.float32) / 255.0  
        img = np.expand_dims(img, axis = 0)

        img = preprocess_images(img, model_type = self.model_type, device=self.device)

        keypoint = self.keypoints[idx]

        keypoint = torch.tensor(keypoint, dtype=torch.float32).to(self.device)

        # Apply any transformations
        if self.transform:
            img, keypoint = apply_transform(img, keypoint, version=self.transform)
        

        # visualize_image(img[0, 0].cpu().numpy(), points=[tuple(keypoint[0].tolist()), tuple(keypoint[1].tolist()), tuple(keypoint[2].tolist())])

        return img, keypoint


if __name__ == "__main__":
    dataset = 'D:/mmissana/data/dataset/train.npz'
    keypoint_dataset = KeypointDataset(dataset, filter=True)
    print(f"Number of images: {len(keypoint_dataset)}")
    print(f"Image shape: {keypoint_dataset[0][0].shape}")
    print(f"Keypoint shape: {keypoint_dataset[0][1].shape}")
    print(f"Keypoint coordinates: {keypoint_dataset[0][1]}")

    