import torch

def center_of_mass(tensor: torch.Tensor, thresh=0.9, device='cpu', normalize = False):
    if len(tensor.shape) != 2:
        raise ValueError("Input tensor must be 2D")
    
    tensor = tensor.float().to(device)  # Ensure float type and move to the correct device

    # Shift values to be non-negative without breaking computation graph
    min_val = tensor.min()
    clipped_tensor = tensor - min_val

    # Apply thresholding in a differentiable way
    threshold = clipped_tensor.max() * thresh
    clipped_tensor = torch.where(clipped_tensor >= threshold, clipped_tensor - threshold, torch.tensor(0.0, device=device, dtype=tensor.dtype))

    total_mass = clipped_tensor.sum()

    # If the total mass is zero, return the geometric center
    if total_mass == 0:
        return torch.tensor([tensor.shape[1] / 2, tensor.shape[0] / 2], device=device, dtype=tensor.dtype, requires_grad=True)

    height, width = tensor.shape

    # Create coordinate tensors on the same device
    x_coords = torch.arange(width, device=device, dtype=tensor.dtype).repeat(height, 1)
    y_coords = torch.arange(height, device=device, dtype=tensor.dtype).repeat(width, 1).t()

    # Compute weighted sum of coordinates
    x_center = (clipped_tensor * x_coords).sum() / total_mass
    y_center = (clipped_tensor * y_coords).sum() / total_mass

    if normalize:
        x_center = x_center / tensor.shape[1]
        y_center = y_center / tensor.shape[0]

    # Return a differentiable tensor
    return torch.stack([x_center, y_center])


def center_of_mass_3d(tensor: torch.Tensor, thresh=0.7, device='cpu', normalize=False):
    """
    Compute the center of mass for a 4D tensor of shape [B, C, H, W].
    Returns a tensor of shape [B, C, 2] with (x, y) coordinates.
    """
    if tensor.ndim != 4:
        raise ValueError("Input tensor must be 4D (B, C, H, W)")

    tensor = tensor.float().to(device)

    B, C, H, W = tensor.shape
    min_val = tensor.view(B, C, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    clipped = tensor - min_val  # shift min to zero

    max_val = clipped.view(B, C, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    threshold = max_val * thresh
    clipped = torch.where(clipped >= threshold, clipped - threshold, torch.zeros_like(clipped))

    total_mass = clipped.sum(dim=(2, 3), keepdim=True)  # shape: [B, C, 1, 1]

    # Avoid division by zero: use geometric center if total mass is zero
    eps = 1e-6
    total_mass = total_mass + eps  # to avoid divide-by-zero

    # Coordinate grids
    y_coords = torch.arange(H, device=device, dtype=tensor.dtype).view(1, 1, H, 1).expand(B, C, H, W)
    x_coords = torch.arange(W, device=device, dtype=tensor.dtype).view(1, 1, 1, W).expand(B, C, H, W)

    x_center = (clipped * x_coords).sum(dim=(2, 3), keepdim=False) / total_mass.squeeze(-1).squeeze(-1)
    y_center = (clipped * y_coords).sum(dim=(2, 3), keepdim=False) / total_mass.squeeze(-1).squeeze(-1)

    if normalize:
        x_center = x_center / W
        y_center = y_center / H

    # Stack into shape [B, C, 2]
    return torch.stack([x_center, y_center], dim=-1)
