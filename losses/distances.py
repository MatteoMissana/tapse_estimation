import torch
import torch.nn as nn


class UnorderedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Custom loss function to handle unordered pairs of points.
        It computes the MSE loss considering both possible matchings and 
        takes the minimum loss for each sample in the batch.

        :param reduction: Specifies the reduction to apply to the output.
                          'mean' (default) computes the average loss.
                          'sum' computes the sum of all losses.
                          'none' returns loss per sample.
        """
        super(UnorderedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss_fn = nn.MSELoss(reduction='none')

        loss1 = loss_fn(pred, target).mean(dim=(1, 2))
        loss2 = loss_fn(pred, target.flip(dims=[1])).mean(dim=(1, 2))

        min_loss = torch.where(loss1 < loss2, loss1, loss2)  # Keeps gradients!

        if self.reduction == 'mean':
            return min_loss.mean()
        elif self.reduction == 'sum':
            return min_loss.sum()
        else:
            return min_loss  # Preserves gradients


def pairwise_distance(a, b):
    """
    Computes the Euclidean distance between corresponding points in two tensors.
    Assumes input shapes are (batch_size, num_points, dim).
    """
    return torch.norm(a - b, dim=2)


class UnorderedDistanceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Custom loss function to handle unordered pairs of points.
        It computes the Euclidean distance considering both possible matchings
        and takes the minimum loss for each sample in the batch.

        :param reduction: Specifies the reduction to apply to the output.
                          'mean' (default) computes the average loss.
                          'sum' computes the sum of all losses.
                          'none' returns loss per sample.
        """
        super(UnorderedDistanceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        dist1 = pairwise_distance(pred, target).mean(dim=1)  # Standard pairing
        dist2 = pairwise_distance(pred, target.flip(dims=[1])).mean(dim=1)  # Flipped pairing

        min_loss = torch.minimum(dist1, dist2)  # Selects the lower distance for each sample

        if self.reduction == 'mean':
            return min_loss.mean()
        elif self.reduction == 'sum':
            return min_loss.sum()
        else:
            return min_loss  # Preserves gradients


# Example usage
if __name__ == "__main__":
    batch_size = 4
    num_classes = 2  # Assuming each sample has exactly 2 points

    pred = torch.rand(batch_size, num_classes, 2)  # Random predicted points
    target = torch.rand(batch_size, num_classes, 2)  # Random ground-truth points

    loss_fn = UnorderedMSELoss(reduction='mean')  # Create loss criterion
    loss = loss_fn(pred, target)  # Compute loss
    print("Loss:", loss.item())
