import torch
import torchvision.transforms.functional as TF

def random_h_flip(image, keypoints, p=0.5):
    """Randomly flips the image horizontally and adjusts keypoints."""
    if torch.rand(1).item() < p:
        image = TF.hflip(image)  # Optimized horizontal flip
        keypoints[:, 0] = image.shape[-1] - keypoints[:, 0]  # Correct x-coordinate adjustment
    return image, keypoints

def adjust_brightness_contrast(image_sequence, brightness_range=(0.8, 1.2), contrast_range=(0.7, 1.3)):
    """
    Adjusts brightness and contrast for a batch of grayscale frames on GPU.
    Input shape: [T, H, W] (torch.Tensor on GPU)
    Output shape: [T, H, W]
    """
    T, H, W = image_sequence.shape
    device = image_sequence.device

    # Add channel dim -> [T, 1, H, W]
    image_sequence = image_sequence.unsqueeze(1)

    # Random brightness and contrast factors on GPU
    brightness_factor = torch.empty(1).uniform_(*brightness_range).item()
    contrast_factor = torch.empty(1).uniform_(*contrast_range).item()

    image_sequence = TF.adjust_brightness(image_sequence, brightness_factor)
    image_sequence = TF.adjust_contrast(image_sequence, contrast_factor)

    image_sequence = image_sequence.squeeze(1)

    return torch.clamp(image_sequence, 0.0, 1.0)  

def random_rotate(image_sequence, keypoints, degrees=(-30, 30), p=0.5):
    """
    Randomly rotates a volume [T, H, W] and adjusts keypoints accordingly.
    Rotation is applied consistently across all frames.
    
    - image_sequence: Tensor [T, H, W], grayscale frames on GPU.
    - keypoints: Tensor [N, 2] or compatible shape (x, y) coords.
    - degrees: Rotation angle range.
    - p: Probability of applying rotation.
    
    Returns:
        rotated_sequence: [T, H, W]
        rotated_keypoints: same shape as input
    """
    if torch.rand(1, device=image_sequence.device) < p:
        angle = torch.empty(1, device=image_sequence.device).uniform_(*degrees).item()
        T, H, W = image_sequence.shape
        center = torch.tensor([W / 2, H / 2], device=image_sequence.device)

        # Rotate each frame with the same angle
        rotated_frames = [
            TF.rotate(image_sequence[i].unsqueeze(0), angle).squeeze(0)
            for i in range(T)
        ]
        rotated_sequence = torch.stack(rotated_frames)

        # Convert angle to radians
        angle_rad = torch.deg2rad(torch.tensor(angle, device=image_sequence.device))

        # Rotation matrix (clockwise for image, so reverse angle for points)
        rot_mat = torch.tensor([
            [torch.cos(-angle_rad), -torch.sin(-angle_rad)],
            [torch.sin(-angle_rad),  torch.cos(-angle_rad)]
        ], device=image_sequence.device)

        rotated_keypoints = keypoints.view(-1, 2)
        rotated_keypoints = (rotated_keypoints - center) @ rot_mat.T + center

        return rotated_sequence, rotated_keypoints.view_as(keypoints)
    else:
        return image_sequence, keypoints

def random_crop(image, keypoints, crop_size = 220, p=0.5):
    """Randomly crops the image, resizes it back, and adjusts keypoints accordingly."""
    if torch.rand(1).item() < p:
        h, w = image.shape[-2], image.shape[-1]
        crop_h = torch.randint(crop_size, image.shape[-2] + 1, (1,)).item()
        crop_w = torch.randint(crop_size, image.shape[-1] + 1, (1,)).item()


        if h <= crop_h or w <= crop_w:
            return image, keypoints  # Skip cropping if image is too small

        # Randomly select the top-left corner of the crop
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()

        # Crop the image
        image = TF.crop(image, top, left, crop_h, crop_w)

        # Resize image back to original size
        image = TF.resize(image, (h, w))

        # Adjust keypoints
        cropped_keypoints = keypoints.view(-1, 2)  # Ensure keypoints are in (N, 2) format
        cropped_keypoints = cropped_keypoints - torch.tensor([left, top], device = cropped_keypoints.device)  # Shift keypoints

        # Scale keypoints to match resized image
        scale_x = w / crop_w
        scale_y = h / crop_h
        cropped_keypoints = cropped_keypoints * torch.tensor([scale_x, scale_y], device = cropped_keypoints.device)

        # Keep only keypoints that remain within the resized area
        # mask = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < w) & \
        #        (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h)
        # keypoints = keypoints[mask]
    
        return image, cropped_keypoints.view_as(keypoints)
    else:
        return image, keypoints

def add_gaussian_noise(image, std=0.02, p=0.5):
    """Adds Gaussian noise with given standard deviation, efficiently."""
    if torch.rand(1, device=image.device).item() < p:
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0.0, 1.0)
    else:
        return image
    
def time_flip(image, keypoints, p=0.5):
    """Randomly flips the time axis of the image and adjusts keypoints."""
    if torch.rand(1).item() < p:
        image = image.flip(0)  # Flip along the time axis (first dimension)
        keypoints = keypoints.flip(0)  # Flip keypoints accordingly
    return image, keypoints

def apply_transform(image: torch.Tensor, keypoints: torch.Tensor, version: str = '0'):
    """Apply transformations to an image and its keypoints."""
    if version == '0':
        pass
    
    elif version == '1':
        image, keypoints = random_h_flip(image, keypoints, p=0.5)
        image = adjust_brightness_contrast(image, contrast_range=(0.8, 1.2))
        image = add_gaussian_noise(image)

    elif version == '2':
        image = adjust_brightness_contrast(image, brightness_range=(0.5, 1.4), contrast_range=(0.5, 1.4))
        image = add_gaussian_noise(image)
    
    elif version == '3':
        image = adjust_brightness_contrast(image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3))
        image = add_gaussian_noise(image, std = 0.02)
    elif version == '4':
        image = adjust_brightness_contrast(image, brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5))
        image = add_gaussian_noise(image, std = 0.08)
        image, keypoints = random_rotate(image, keypoints, p=1)
    elif version == '5':
        image = adjust_brightness_contrast(image, brightness_range=(0.6, 1.4), contrast_range=(0.6, 1.4))
        image = add_gaussian_noise(image, std = 0.06)
        image, keypoints = random_rotate(image, keypoints, p=.6)
        
    elif version == '6':
        image = adjust_brightness_contrast(image, brightness_range=(0.6, 1.4), contrast_range=(0.6, 1.4))
        image = add_gaussian_noise(image, std = 0.06)
        image, keypoints = random_rotate(image, keypoints, p=.5)
        image, keypoints = random_crop(image, keypoints, crop_size=220, p=.6)
    elif version == '7':
        image = adjust_brightness_contrast(image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2))
        image = add_gaussian_noise(image, std = 0.04)
        image, keypoints = random_rotate(image, keypoints, degrees=(-15, 15), p=.6)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=.6)
    elif version == '8':
        # image = adjust_brightness_contrast(image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2))
        image = add_gaussian_noise(image, std = 0.04)
        image, keypoints = random_rotate(image, keypoints, degrees=(-15, 15), p=.6)
        image, keypoints = random_crop(image, keypoints, crop_size=230, p=.6)
        image, keypoints = time_flip(image, keypoints, p=.5)
    else:
        raise ValueError(f"Unsupported version: {version}")

    return image, keypoints