import torch

state_dict = torch.load("dl_mapse/Data/best_true_weights.pth", map_location="cpu")

if isinstance(state_dict, dict):
    print("This is a state_dict (weights only). Keys:", state_dict.keys())
else:
    print("This is a full model.")