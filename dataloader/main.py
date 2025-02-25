import torch
from torch.utils.data import Dataset
import numpy as np
import os

class KeypointDataset(Dataset):
    def __init__(self, numpy_dataset, transform=None, filter=False):
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
            unannotated_indices = np.where(np.all(keypoints == 0, axis=1))[0]

            images = np.delete(images, unannotated_indices, axis=0)
            keypoints = np.delete(keypoints, unannotated_indices, axis=0)

        self.images = images
        self.keypoints = keypoints
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Ensure image is in grayscale format (H, W) → (1, H, W)
        if img.ndim == 2:  # Grayscale image (H, W)
            img = np.expand_dims(img, axis=0)  # Convert to (1, H, W)

        # Normalize image to [0,1]
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img)  # Already in (1, H, W)

        # Normalize keypoints based on image dimensions
        # keypoints = list(self.keypoints[idx])
        keypoint = self.keypoints[idx]
        keypoint[0, 0] /= img.shape[2]  # x1 / width
        keypoint[0, 1] /= img.shape[1]  # y1 / height
        keypoint[1, 0] /= img.shape[2]  # x2 / width
        keypoint[1, 1] /= img.shape[1]  # y2 / height

        keypoint = torch.tensor(keypoint.flatten(), dtype=torch.float32)

        # Apply any transformations
        if self.transform:
            img, keypoint = self.transform(img, keypoint)

        return img, keypoint


if __name__ == "__main__":
    dataset = 'D:/mmissana/data/dataset/dataset.npz'
    keypoint_dataset = KeypointDataset(dataset, filter=True)
    print(f"Number of images: {len(keypoint_dataset)}")
    print(f"Image shape: {keypoint_dataset[0][0].shape}")
    print(f"Keypoint shape: {keypoint_dataset[0][1].shape}")
    print(f"Keypoint coordinates: {keypoint_dataset[0][1]}")
    
    