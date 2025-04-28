import cupy as cp
import numpy as np
import torch


def tric_apex_distance_calculation(free_wall, septum, apex):
    middle = (free_wall + septum)/2

    dist_0 = torch.linalg.vector_norm(free_wall - apex, ord=2)
    dist_1 = torch.linalg.vector_norm(septum - apex, ord=2)
    dist_2 = torch.linalg.vector_norm(middle - apex, ord=2)

    dist = (dist_0 + dist_1 + dist_2) / 3
    dist = dist.cpu().numpy()
    return dist