import torch
from torch.utils.data import Dataset
import numpy as np
import os
from dataloader.preprocessing import preprocess_images
from torchvision import transforms as T
from torchvision.transforms import v2 as T
from augmentations.aug_memory import apply_transform
from utils.plot import visualize_image


def create_gaussian_kernel(size, std):
    """
    Crea un template 2D gaussiano centrato in (size//2, size//2).
    """
    ax = torch.arange(size).float()
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    center = size // 2
    kernel = torch.exp(-((xx - center)**2 + (yy - center)**2) / (2 * std**2))
    return kernel

def generate_image_with_gaussians(image_size, points, std, kernel_size=None):
    """
    Inserisce gaussiane precompute nei punti specificati.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.zeros((image_size, image_size), device=device)

    if kernel_size is None:
        # default: 6 sigma window
        kernel_size = int(6 * std)
        if kernel_size % 2 == 0:
            kernel_size += 1

    kernel = create_gaussian_kernel(kernel_size, std).to(device)
    half_k = kernel_size // 2

    for x, y in points[0]:
        x = int(round(x))
        y = int(round(y))

        # Definisce i bounds della zona da aggiornare nell'immagine
        x_start = max(x - half_k, 0)
        y_start = max(y - half_k, 0)
        x_end = min(x + half_k + 1, image_size)
        y_end = min(y + half_k + 1, image_size)

        # Bounds della zona nel kernel
        kx_start = half_k - (x - x_start)
        ky_start = half_k - (y - y_start)
        kx_end = kx_start + (x_end - x_start)
        ky_end = ky_start + (y_end - y_start)

        # Somma la parte della gaussiana
        image[y_start:y_end, x_start:x_end] += kernel[ky_start:ky_end, kx_start:kx_end]

    image = torch.clamp(image, 0.0, 1.0)
    return image.unsqueeze(0)  # (1, H, W)


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

        gaussian_map = generate_image_with_gaussians(256, [self.keypoints[idx-1].tolist()], std=10.0).to(self.device)
        gaussian_map = gaussian_map.unsqueeze(0)
        # visualize_image(gaussian_map[0].cpu().numpy(), points=[tuple(keypoint[0].tolist()), tuple(keypoint[1].tolist()), tuple(keypoint[2].tolist())])

        # Apply any transformations
        if self.transform:
            img, gaussian_map, keypoint = apply_transform(img, gaussian_map, keypoint, version=self.transform)
            # gaussian_map, _ = apply_transform(gaussian_map, keypoint, version=self.transform)
        
        visualize_image(img[0, 0].cpu().numpy(), points=[tuple(keypoint[0].tolist()), tuple(keypoint[1].tolist()), tuple(keypoint[2].tolist())])
        visualize_image(gaussian_map[0, 0].cpu().numpy(), points=[tuple(keypoint[0].tolist()), tuple(keypoint[1].tolist()), tuple(keypoint[2].tolist())])

        input = torch.cat((img, gaussian_map), dim=1)

        return input, keypoint


if __name__ == "__main__":
    dataset = r'D:\mmissana\data\RV_PATIENTS\dataset_256\train.npz'
    keypoint_dataset = KeypointDataset(dataset, filter=True)
    print(f"Number of images: {len(keypoint_dataset)}")
    print(f"Image shape: {keypoint_dataset[0][0].shape}")
    print(f"Keypoint shape: {keypoint_dataset[0][1].shape}")
    print(f"Keypoint coordinates: {keypoint_dataset[0][1]}")

