import torch
import torchvision.transforms.functional as TF

def random_h_flip(img1, img2, keypoints, p=0.5):
    if torch.rand(1).item() < p:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)
        w = img1.shape[-1]
        keypoints[:, 0] = w - keypoints[:, 0]
    return img1, img2, keypoints

def adjust_brightness_contrast(image, brightness_range=(0.8, 1.2), contrast_range=(0.7, 1.3)):
    brightness_factor = torch.empty(1).uniform_(*brightness_range).item()
    contrast_factor = torch.empty(1).uniform_(*contrast_range).item()
    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)
    return image

def random_rotate(img1, img2, keypoints, degrees=(-30, 30), p=0.5):
    if torch.rand(1).item() < p:
        angle = torch.empty(1).uniform_(*degrees).item()
        h, w = img1.shape[-2], img1.shape[-1]
        center = torch.tensor([w / 2, h / 2])

        img1 = TF.rotate(img1, angle)
        img2 = TF.rotate(img2, angle)

        angle_rad = torch.deg2rad(torch.tensor(angle))
        rot_matrix = torch.tensor([
            [torch.cos(-angle_rad), -torch.sin(-angle_rad)],
            [torch.sin(-angle_rad), torch.cos(-angle_rad)]
        ])

        keypoints = (keypoints - center) @ rot_matrix.T + center

    return img1, img2, keypoints

def random_crop(img1, img2, keypoints, crop_size=220, p=0.5):
    if torch.rand(1).item() < p:
        h, w = img1.shape[-2], img1.shape[-1]
        crop_h = torch.randint(crop_size, h + 1, (1,)).item()
        crop_w = torch.randint(crop_size, w + 1, (1,)).item()

        if h <= crop_h or w <= crop_w:
            return img1, img2, keypoints

        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()

        img1 = TF.crop(img1, top, left, crop_h, crop_w)
        img2 = TF.crop(img2, top, left, crop_h, crop_w)

        img1 = TF.resize(img1, (h, w))
        img2 = TF.resize(img2, (h, w))

        shift = torch.tensor([left, top])
        scale = torch.tensor([w / crop_w, h / crop_h])
        keypoints = (keypoints - shift) * scale

    return img1, img2, keypoints

def add_gaussian_noise(image, std=0.02):
    noise = torch.normal(0, std, size=image.shape, device=image.device)
    return torch.clamp(image + noise, 0.0, 1.0)

def slightly_perturb_image(image, degrees=(-5, 5), translate=(0.02, 0.02), crop_size=None):
    """Applies minor spatial jittering to the image without affecting keypoints."""
    # Random rotation
    angle = torch.empty(1).uniform_(*degrees).item()
    image = TF.rotate(image, angle)

    # Random translation (percentage of image size)
    h, w = image.shape[-2:]
    max_dx = int(translate[0] * w)
    max_dy = int(translate[1] * h)
    translate_x = torch.randint(-max_dx, max_dx + 1, (1,)).item()
    translate_y = torch.randint(-max_dy, max_dy + 1, (1,)).item()
    image = TF.affine(image, angle=0, translate=[translate_x, translate_y], scale=1.0, shear=[0.0, 0.0])

    # Optional small crop
    if crop_size is not None:
        crop_h = crop_w = crop_size
        if h > crop_h and w > crop_w:
            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()
            image = TF.crop(image, top, left, crop_h, crop_w)
            image = TF.resize(image, (h, w))
    return image

def apply_transform(img1: torch.Tensor, img2: torch.Tensor, keypoints: torch.Tensor, version: str = '0'):
    if version == '0':
        pass
    elif version == '1':
        img1, img2, keypoints = random_h_flip(img1, img2, keypoints)
        img1 = adjust_brightness_contrast(img1, contrast_range=(0.8, 1.2))
        img1 = add_gaussian_noise(img1)
    elif version == '2':
        img1 = adjust_brightness_contrast(img1, (0.7, 1.3), (0.7, 1.3))
        img1 = add_gaussian_noise(img1)
    elif version == '3':
        img1 = adjust_brightness_contrast(img1, (0.5, 1.5), (0.5, 1.5))
        img1 = add_gaussian_noise(img1, 0.08)
    elif version == '4':
        img1 = adjust_brightness_contrast(img1, (0.5, 1.5), (0.5, 1.5))
        img1 = add_gaussian_noise(img1, 0.08)
        img1, img2, keypoints = random_rotate(img1, img2, keypoints, p=1)
    elif version == '5':
        img1 = adjust_brightness_contrast(img1, (0.6, 1.4), (0.6, 1.4))
        img1 = add_gaussian_noise(img1, 0.06)
        img1, img2, keypoints = random_rotate(img1, img2, keypoints, p=0.6)
    elif version == '6':
        img1 = adjust_brightness_contrast(img1, (0.6, 1.4), (0.6, 1.4))
        img1 = add_gaussian_noise(img1, 0.06)
        img1, img2, keypoints = random_rotate(img1, img2, keypoints, p=0.5)
        img1, img2, keypoints = random_crop(img1, img2, keypoints, crop_size=220, p=0.6)
    elif version == '7':
        img1 = adjust_brightness_contrast(img1, (0.8, 1.2), (0.8, 1.2))
        img1 = add_gaussian_noise(img1, 0.04)
        img1, img2, keypoints = random_rotate(img1, img2, keypoints, degrees=(-15, 15), p=0.6)
        img1, img2, keypoints = random_crop(img1, img2, keypoints, crop_size=230, p=0.6)
        img2 = slightly_perturb_image(img2, degrees=(-3, 3), translate=(0.001, 0.001), crop_size=240)

    else:
        raise ValueError(f"Unsupported version: {version}")

    return img1, img2, keypoints
