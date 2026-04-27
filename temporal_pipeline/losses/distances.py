import torch
import torch.nn as nn



class OrderedDistanceLoss_3d(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Custom loss function for ordered keypoints.
        Computes the Euclidean distance between corresponding keypoints in 'pred' and 'target'.
        Handles inputs of shape (B, N, 3, 2) or (B, 3, 2).
        """
        super(OrderedDistanceLoss_3d, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        :param pred: Tensor of predicted keypoints. Shape: (B, N, 3, 2) or (B, 3, 2)
        :param target: Tensor of ground-truth keypoints. Same shape as pred.
        :return: Loss value (scalar or per sample)
        """
        # Ensure pred and target are 3D: (batch_size * N, 3, 2)
        # print(f"pred shape: {pred.shape}")
        if pred.dim() == 4:
            B, N, P, C = pred.shape  # Usually (1, 64, 3, 2)
            pred = pred.reshape(-1, P, C)   # Shape: (B*N, 3, 2)
            target = target.reshape(-1, P, C)
        elif pred.dim() == 3:
            # Already in shape (B, 3, 2), nothing to change
            pass
        else:
            raise ValueError(f"Unexpected input shape: {pred.shape}")
        
        # print(f"pred shape after: {pred.shape}")


        # Compute Euclidean distances
        distances = torch.linalg.norm(pred - target, ord = 2, dim=2)  # Shape: (B*N, 3)

        # Reduce
        if self.reduction == 'mean':
            return distances.mean()
        elif self.reduction == 'sum':
            return distances.sum()
        elif self.reduction == 'max':
            return distances.max()
        else:  # 'none'
            return distances


class CombinedLandmarkLoss(nn.Module):
    """
    Combined loss for video landmark tracking.

    Addresses the 'static shortcut' problem where the model predicts
    the temporal mean position of each landmark instead of tracking it.

    Components:
        1. dist_loss   — standard L2 distance per frame per landmark
        2. motion_loss — supervises frame-to-frame *changes* in position
        3. var_loss    — penalizes predictions that are too static
                         compared to ground-truth motion range

    Args:
        lambda_motion (float): weight for motion_loss. Default 0.5.
        lambda_var    (float): weight for var_loss.    Default 0.3.
        reduction     (str):   'mean' or 'sum'.        Default 'mean'.

    Input shapes:
        Previously : [B, N, 3, 2]   (B=batch, N=frames)
        Now accepts : [B, N, 3, 2]  unchanged — but also supports an
                      extra leading sample dimension coming from the
                      updated center_of_mass_3d output [N_batch, N, 3, 2],
                      where N_batch is the true batch size.
        In practice the forward() signature is unchanged; callers simply
        pass larger B values when batching.
    """

    def __init__(self, lambda_motion=1, lambda_var=0.1, reduction='mean'):
        super().__init__()
        self.lambda_motion = lambda_motion
        self.lambda_var    = lambda_var
        self.reduction     = reduction

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the chosen reduction to a tensor of arbitrary shape."""
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        return x  # 'none' — return unreduced tensor

    # ------------------------------------------------------------------
    # loss components
    # ------------------------------------------------------------------

    def _dist_loss(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard per-frame, per-landmark L2 distance loss.

        Args:
            pred, target: [B*N, 3, 2]
                          B = total samples in the batch (may include
                              multiple videos), N = frames, already
                              merged into a single leading axis by the
                              caller so every (sample, frame) pair is
                              treated independently.

        Returns:
            Scalar loss.
        """
        # Euclidean distance per landmark per (sample × frame)
        # distances shape: [B*N, 3]
        distances = torch.linalg.norm(pred - target, ord=2, dim=-1)
        return self._reduce(distances)

    def _motion_loss(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supervise frame-to-frame *motion*, not just absolute position.

        The model is penalised if its predicted displacement between
        consecutive frames differs from the ground-truth displacement.
        This directly combats the 'predict the mean' shortcut, because
        the mean is constant (zero displacement every frame), which
        will always produce large errors here whenever the landmark
        is actually moving.

        Args:
            pred, target: [B, N, 3, 2]
                          B = batch size (number of video clips),
                          N = frames per clip,
                          3 = landmarks, 2 = (x, y) coordinates.

        Returns:
            Scalar loss.
        """
        # Frame-to-frame displacement vectors — shape: [B, N-1, 3, 2]
        pred_delta   = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_delta = target[:, 1:, :, :] - target[:, :-1, :, :]

        # L2 error between predicted and target displacement — [B, N-1, 3]
        motion_error = torch.linalg.norm(
            pred_delta - target_delta, ord=2, dim=-1
        )
        return self._reduce(motion_error)

    def _var_loss(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalise predictions whose temporal spread is smaller than
        the ground-truth temporal spread.

        If the model predicts nearly constant coordinates over N frames
        (variance ≈ 0), but the target landmarks actually move
        (variance > 0), this term fires and pushes the model to spread
        its predictions out.

        relu(target_var - pred_var) is 0 when the model already covers
        the target's motion range, and positive only when it falls short.
        This is an asymmetric penalty: the model is never pushed to be
        *more* variable than the target, only *at least* as variable.

        Args:
            pred, target: [B, N, 3, 2]
                          Variance is computed across the N (frame)
                          dimension independently for every sample in B.

        Returns:
            Scalar loss.
        """
        # Temporal variance per (batch, landmark, coord) — shape: [B, 3, 2]
        pred_var   = pred.var(dim=1)
        target_var = target.var(dim=1)

        # Only penalise when predicted variance falls *below* target variance
        var_deficit = torch.relu(target_var - pred_var)
        return self._reduce(var_deficit)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred   (Tensor): predicted landmarks.
                             Accepted shapes:
                               [B, N, 3, 2]  — standard batched input
                               [N, 3, 2]     — single sample (auto-unsqueezed)
            target (Tensor): ground-truth landmarks, same shape as pred.

        Returns:
            total_loss (Tensor): scalar, backprop-ready.
            breakdown  (dict):   {'dist', 'motion', 'var'} for logging.
        """
        # ── rank normalisation ──────────────────────────────────────────
        # Accept a single un-batched sample and promote it to [1, N, 3, 2]
        if pred.dim() == 3:
            pred   = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # ── validate ────────────────────────────────────────────────────
        if pred.dim() != 4 or pred.shape[-2:] != torch.Size([3, 2]):
            raise ValueError(
                f"Expected pred shape [B, N, 3, 2] (or [N, 3, 2]), "
                f"got {pred.shape}"
            )
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target shapes must match: "
                f"{pred.shape} vs {target.shape}"
            )

        # B = batch size (number of video clips in this mini-batch)
        # N = number of frames per clip
        # P = number of landmarks (3)
        # C = coordinate dims   (2 → x, y)
        B, N, P, C = pred.shape

        # ── component 1: frame-level distance ───────────────────────────
        # Merge the batch and frame axes so that every (clip, frame) pair
        # is scored independently as a set of 3 landmarks.
        # Shape: [B, N, 3, 2] → [B*N, 3, 2]
        pred_flat   = pred.reshape(B * N, P, C)
        target_flat = target.reshape(B * N, P, C)
        loss_dist   = self._dist_loss(pred_flat, target_flat)

        # ── component 2: motion consistency ─────────────────────────────
        # Needs the temporal axis intact to compute Δ between frames.
        # Shape stays: [B, N, 3, 2]
        loss_motion = self._motion_loss(pred, target)

        # ── component 3: temporal variance penalty ───────────────────────
        # Needs the temporal axis intact to compute per-clip variance.
        # Shape stays: [B, N, 3, 2]
        loss_var = self._var_loss(pred, target)

        # ── combine ──────────────────────────────────────────────────────
        total_loss = (
            loss_dist
            + self.lambda_motion * loss_motion
            + self.lambda_var    * loss_var
        )

        # Breakdown dict — weighted values for TensorBoard / W&B logging
        breakdown = {
            'dist':   loss_dist.item(),
            'motion': loss_motion.item() * self.lambda_motion,
            'var':    loss_var.item()    * self.lambda_var,
        }

        return total_loss, breakdown

# Example usage
if __name__ == "__main__":
    batch_size = 4
    num_classes = 2  # Assuming each sample has exactly 2 points

    pred = torch.rand(batch_size, num_classes, 2)  # Random predicted points
    target = torch.rand(batch_size, num_classes, 2)  # Random ground-truth points

    loss_fn = UnorderedMSELoss(reduction='mean')  # Create loss criterion
    loss = loss_fn(pred, target)  # Compute loss
    print("Loss:", loss.item())
