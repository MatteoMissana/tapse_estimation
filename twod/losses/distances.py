import torch
import torch.nn as nn


# class UnorderedMSELoss(nn.Module):
#     def __init__(self, reduction='mean'):
#         """
#         Custom loss function to handle unordered pairs of points.
#         It computes the MSE loss considering both possible matchings and 
#         takes the minimum loss for each sample in the batch.

#         :param reduction: Specifies the reduction to apply to the output.
#                           'mean' (default) computes the average loss.
#                           'sum' computes the sum of all losses.
#                           'none' returns loss per sample.
#         """
#         super(UnorderedMSELoss, self).__init__()
#         self.reduction = reduction

#     def forward(self, pred, target):
#         loss_fn = nn.MSELoss(reduction='none')

#         loss1 = loss_fn(pred, target).mean(dim=(1, 2))
#         loss2 = loss_fn(pred, target.flip(dims=[1])).mean(dim=(1, 2))

#         min_loss = torch.where(loss1 < loss2, loss1, loss2)  # Keeps gradients!

#         if self.reduction == 'mean':
#             return min_loss.mean()
#         elif self.reduction == 'sum':
#             return min_loss.sum()
#         else:
#             return min_loss  # Preserves gradients


# def pairwise_distance(a, b):
#     """
#     Computes the Euclidean distance between corresponding points in two tensors.
#     Assumes input shapes are (batch_size, num_points, dim).
#     """
#     return torch.norm(a - b, dim=2)


# class UnorderedDistanceLoss(nn.Module):
#     def __init__(self, reduction='mean'):
#         """
#         Custom loss function to handle unordered pairs of points.
#         It computes the Euclidean distance considering both possible matchings
#         and takes the minimum loss for each sample in the batch.

#         :param reduction: Specifies the reduction to apply to the output.
#                           'mean' (default) computes the average loss.
#                           'sum' computes the sum of all losses.
#                           'none' returns loss per sample.
#         """
#         super(UnorderedDistanceLoss, self).__init__()
#         self.reduction = reduction

#     def forward(self, pred, target):
#         dist1 = pairwise_distance(pred, target).mean(dim=1)  # Standard pairing
#         dist2 = pairwise_distance(pred, target.flip(dims=[1])).mean(dim=1)  # Flipped pairing

#         min_loss = torch.minimum(dist1, dist2)  # Selects the lower distance for each sample

#         if self.reduction == 'mean':
#             return min_loss.mean()
#         elif self.reduction == 'sum':
#             return min_loss.sum()
#         else:
#             return min_loss  # Preserves gradients
        
class OrderedDistanceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Custom loss function for ordered keypoints.
        Computes the Euclidean distance between corresponding keypoints in 'pred' and 'target'.
        Handles inputs of shape (B, N, 3, 2) or (B, 3, 2).
        """
        super(OrderedDistanceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        :param pred: Tensor of predicted keypoints. Shape: (B, N, 3, 2) or (B, 3, 2)
        :param target: Tensor of ground-truth keypoints. Same shape as pred.
        :return: Loss value (scalar or per sample)
        """
        # Ensure pred and target are 3D: (batch_size * N, 3, 2)
        if pred.dim() == 4:
            B, N, P, C = pred.shape  # Usually (1, 64, 3, 2)
            pred = pred.view(-1, P, C)   # Shape: (B*N, 3, 2)
            target = target.view(-1, P, C)
        elif pred.dim() == 3:
            # Already in shape (B, 3, 2), nothing to change
            pass
        else:
            raise ValueError(f"Unexpected input shape: {pred.shape}")

        # Compute Euclidean distances
        distances = torch.norm(pred - target, dim=2)  # Shape: (B*N, 3)

        # Reduce
        if self.reduction == 'mean':
            return distances.mean()
        elif self.reduction == 'sum':
            return distances.sum()
        else:  # 'none'
            return distances


class GaussianKeypointLoss(nn.Module):
    def __init__(self, sigma=1.0, reduction='mean'):
        """
        Loss basata su una gaussiana centrata nel punto reale.
        Loss = 1 - exp(-||pred - target||^2 / (2 * sigma^2))
        """
        super(GaussianKeypointLoss, self).__init__()
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, pred, target):
        if pred.dim() == 4:
            B, N, P, C = pred.shape
            pred = pred.view(-1, P, C)
            target = target.view(-1, P, C)
        elif pred.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {pred.shape}")

        # Calcolo della distanza quadratica (||x - y||^2)
        sq_dist = torch.sum((pred - target) ** 2, dim=2)  # Shape: (B*N, 3)

        # Calcolo del valore della gaussiana
        gaussian = torch.exp(-sq_dist / (2 * self.sigma ** 2))  # max 1, decresce con la distanza

        loss = 1.0 - gaussian  # più lontano => loss più alta

        # Riduzione
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# Example usage
if __name__ == "__main__":
    batch_size = 4
    num_classes = 2  # Assuming each sample has exactly 2 points

    pred = torch.rand(batch_size, num_classes, 2)  # Random predicted points
    target = torch.rand(batch_size, num_classes, 2)  # Random ground-truth points

    loss_fn = UnorderedMSELoss(reduction='mean')  # Create loss criterion
    loss = loss_fn(pred, target)  # Compute loss
    print("Loss:", loss.item())
