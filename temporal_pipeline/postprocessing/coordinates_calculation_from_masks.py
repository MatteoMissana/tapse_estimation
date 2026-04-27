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


def center_of_mass_3d(tensor: torch.Tensor, thresh=0.9, device='cpu', normalize=False):
    """
    Computes the 2D center of mass for each landmark and frame in a batch of tensors.

    Args:
        tensor:    Input tensor of shape [N, 3, 32, H, W] (batched)
                   or [3, 32, H, W] / [1, 3, 32, H, W] (single sample).
                   Axis meaning: N=batch, 3=landmarks, 32=frames, H=height, W=width.
        thresh:    Fraction of the per-channel max used as a soft threshold (default 0.9).
                   Pixels below this fraction are zeroed out before computing the CoM.
        device:    Torch device string, e.g. 'cpu' or 'cuda' (default 'cpu').
        normalize: If True, divides x_center by W and y_center by H so coordinates
                   are in [0, 1] (default False).

    Returns:
        coords:    Float tensor of shape [N, C, B, 2], where
                   N = batch size, C = frames (32), B = landmarks (3), 2 = (x, y).
    """

    # ------------------------------------------------------------------ #
    # 1.  Normalise rank: make sure we always work with a 5-D tensor      #
    #     [N, B, C, H, W]  (N=batch, B=landmarks, C=frames)               #
    # ------------------------------------------------------------------ #
    if tensor.ndim == 4:
        # Single sample supplied without a batch dimension → add one
        tensor = tensor.unsqueeze(0)          # [1, B, C, H, W]

    if tensor.ndim != 5:
        raise ValueError(
            f"Expected a 4-D or 5-D input tensor, got shape {tensor.shape}"
        )

    # Move to the requested device and cast to float32 for all arithmetic
    tensor = tensor.float().to(device)

    N, B, C, H, W = tensor.shape   # unpack all five axes for clarity
    # N = batch size          (e.g. 4)
    # B = number of landmarks (e.g. 3)
    # C = number of frames    (e.g. 32)
    # H, W = spatial dims     (e.g. 256 × 256)

    # ------------------------------------------------------------------ #
    # 2.  Per-channel min-subtraction  →  shift every channel so that     #
    #     its minimum value becomes 0.                                     #
    # ------------------------------------------------------------------ #
    # Collapse H and W into a single axis to find the spatial minimum.
    # Result shape: [N, B, C, 1, 1]  (kept for broadcasting)
    min_val = (tensor
               .view(N, B, C, -1)          # [N, B, C, H*W]
               .min(dim=-1)[0]             # [N, B, C]
               .unsqueeze(-1)              # [N, B, C, 1]
               .unsqueeze(-1))             # [N, B, C, 1, 1]

    clipped = tensor - min_val             # shift min → 0; shape unchanged

    # ------------------------------------------------------------------ #
    # 3.  Soft threshold: zero out everything below `thresh × max`.       #
    #     This focuses the centre-of-mass on the brightest region only.   #
    # ------------------------------------------------------------------ #
    # Per-channel maximum after the min-shift; shape: [N, B, C, 1, 1]
    max_val = (clipped
               .view(N, B, C, -1)
               .max(dim=-1)[0]
               .unsqueeze(-1)
               .unsqueeze(-1))

    threshold = max_val * thresh           # scalar threshold per channel

    # Subtract the threshold and clamp negatives to zero
    clipped = torch.where(
        clipped >= threshold,
        clipped - threshold,               # keep (and shift) values above threshold
        torch.zeros_like(clipped)          # zero out values below threshold
    )

    # ------------------------------------------------------------------ #
    # 4.  Total mass per channel  →  used as the normalising denominator. #
    # ------------------------------------------------------------------ #
    # Sum over the two spatial axes; shape: [N, B, C, 1, 1]
    total_mass = clipped.sum(dim=(3, 4), keepdim=True)

    # "Dead" channels: channels where all pixels were zeroed out.
    # We'll fall back to the image centre for those.
    dead_mask = total_mass < 1e-6          # bool tensor [N, B, C, 1, 1]

    # ------------------------------------------------------------------ #
    # 5.  Coordinate grids  →  one value per pixel indicating its (x, y). #
    # ------------------------------------------------------------------ #
    # y_coords[..., row, :] == row index;  shape → [1, 1, 1, H, 1]
    y_coords = (torch.arange(H, device=device, dtype=tensor.dtype)
                .view(1, 1, 1, H, 1)
                .expand(N, B, C, H, W))

    # x_coords[..., :, col] == col index;  shape → [1, 1, 1, 1, W]
    x_coords = (torch.arange(W, device=device, dtype=tensor.dtype)
                .view(1, 1, 1, 1, W)
                .expand(N, B, C, H, W))

    # ------------------------------------------------------------------ #
    # 6.  Weighted average position  =  centre of mass.                   #
    # ------------------------------------------------------------------ #
    # Replace zero-mass denominators with 1 to avoid NaN (result is       #
    # overwritten by the dead_mask fallback below anyway).
    safe_mass = torch.where(dead_mask, torch.ones_like(total_mass), total_mass)

    # Weighted sum over spatial axes; divide by total mass → [N, B, C, 1, 1]
    x_center = (clipped * x_coords).sum(dim=(3, 4), keepdim=True) / safe_mass
    y_center = (clipped * y_coords).sum(dim=(3, 4), keepdim=True) / safe_mass

    # ------------------------------------------------------------------ #
    # 7.  Dead-channel fallback: use the image centre (W/2, H/2).         #
    # ------------------------------------------------------------------ #
    x_center = torch.where(dead_mask, torch.full_like(x_center, W / 2), x_center)
    y_center = torch.where(dead_mask, torch.full_like(y_center, H / 2), y_center)

    # Remove the two trailing size-1 axes → [N, B, C]
    x_center = x_center.squeeze(-1).squeeze(-1)
    y_center = y_center.squeeze(-1).squeeze(-1)

    # ------------------------------------------------------------------ #
    # 8.  Optional normalisation to [0, 1].                               #
    # ------------------------------------------------------------------ #
    if normalize:
        x_center = x_center / W
        y_center = y_center / H

    # ------------------------------------------------------------------ #
    # 9.  Pack (x, y) pairs and reshape to match expected output layout.  #
    # ------------------------------------------------------------------ #
    # Stack along a new last axis → [N, B, C, 2]
    coords = torch.stack([x_center, y_center], dim=-1)

    # Permute to [N, C, B, 2]  (frames first, then landmarks, then xy)
    # This matches the original single-sample convention [1, 32, 3, 2]
    # but now generalised to a full batch.
    coords = coords.permute(0, 2, 1, 3)   # [N, C, B, 2]

    return coords
