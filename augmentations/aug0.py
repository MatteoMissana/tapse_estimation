import torch
import torchvision.transforms.functional as TF

def random_h_flip(image, keypoints, p=0.5):
    """Randomly flips the image horizontally and adjusts keypoints."""
    if torch.rand(1).item() < p:
        image = TF.hflip(image)  # Optimized horizontal flip
        keypoints[:, 0] = image.shape[-1] - keypoints[:, 0]  # Correct x-coordinate adjustment
    return image, keypoints

def adjust_brightness_contrast(image, brightness_range=(0.8, 1.2), contrast_range=(0.7, 1.3)):
    """Adjusts brightness and contrast with given ranges."""
    brightness_factor = torch.empty(1).uniform_(*brightness_range).item()
    contrast_factor = torch.empty(1).uniform_(*contrast_range).item()
    
    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)
    
    return image

def random_rotate(image, keypoints, degrees=(-30, 30), p=0.5):
    """Randomly rotates the image and adjusts keypoints."""
    if torch.rand(1).item() < p:
        angle = torch.empty(1).uniform_(*degrees).item()  # Random angle in range
        h, w = image.shape[-2], image.shape[-1]
        center = torch.tensor([w / 2, h / 2])

        # Rotate image
        image = TF.rotate(image, angle)

        # Rotate keypoints
        angle_rad = torch.deg2rad(torch.tensor(angle))
        rotation_matrix = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad)],
            [torch.sin(angle_rad), torch.cos(angle_rad)]
        ])

        keypoints = (keypoints[:] - center) @ rotation_matrix.T + center  # Apply rotation

    return image, keypoints

def add_gaussian_noise(image, std=0.02):
    """Adds Gaussian noise with given standard deviation."""
    noise = torch.normal(0, std, size=image.shape, device=image.device)
    return torch.clamp(image + noise, 0.0, 1.0)

def apply_transform(image: torch.Tensor, keypoints: torch.Tensor, version: str = '0'):
    """Apply transformations to an image and its keypoints."""
    if version == '0':
        pass
    
    elif version == '1':
        image, keypoints = random_h_flip(image, keypoints, p=0.5)
        # image, keypoints = random_rotate(image, keypoints)
        image = adjust_brightness_contrast(image, contrast_range=(0.8, 1.2))
        image = add_gaussian_noise(image)

    elif version == '2':
        #image, keypoints = random_h_flip(image, keypoints, p=0.5)
        # image, keypoints = random_rotate(image, keypoints)
        image = adjust_brightness_contrast(image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3))
        image = add_gaussian_noise(image)
    
    elif version == '3':
        #image, keypoints = random_h_flip(image, keypoints, p=0.5)
        # image, keypoints = random_rotate(image, keypoints)
        image = adjust_brightness_contrast(image, brightness_range=(0.6, 1.4), contrast_range=(0.6, 1.4))
        image = add_gaussian_noise(image, std = 0.06)
    else:
        raise ValueError(f"Unsupported version: {version}")

    return image, keypoints
