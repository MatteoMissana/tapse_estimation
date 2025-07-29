import torch
from torch.utils.data import Dataset
import random
import os
import h5py
import numpy as np
from augmentations.augm_3d import apply_transform, apply_transform_val
from utils.plot import visualize_image, VolumeViewer
from dataloader.preprocessing import resize_or_crop_image_torch


class RandomClipDataset(Dataset):
    def __init__(self, videos, keypoints, clip_length=64, transform=None, num_reps=6):
        """
        videos: list of videos, where each video is a tensor of shape (T, C, H, W)
        clip_length: number of frames per clip (e.g., 64)
        transform: optional transform applied to each clip
        """
        self.videos = videos
        self.keypoints = keypoints
        self.clip_length = clip_length
        self.transform = transform
        self.num_reps = num_reps

    def __len__(self):
        return len(self.videos)#*self.num_reps

    def __getitem__(self, idx):
        for i in range(self.num_reps):
            if idx >= (len(self.videos)*i):
                new_idx = idx - (len(self.videos)*i)

        idx = new_idx
        print('index', idx)
        print(self.videos[idx].shape)
        print(self.keypoints[idx].shape)
        video = self.videos[idx]  # Shape: (T, C, H, W)
        T = video.shape[2]

        if T < self.clip_length:
            raise ValueError(f"Video too short: {T} < {self.clip_length}")

        # Scegli inizio clip casuale
        start = random.randint(0, T - self.clip_length - 1)
        end = start + self.clip_length

        clip = video[:, :, start:end]  # Shape: (clip_length, C, H, W)

        keypoints = self.keypoints[idx][start:end]
        

        clip = clip.permute(2,0,1)
        
        if clip.shape[1] != 256 or clip.shape[2] != 256:
            clip, keypoints = resize_or_crop_image_torch(clip, keypoints, target_size=(256, 256))

        if self.transform:
            clip, keypoints = apply_transform(clip, keypoints, version=self.transform)
        clip = clip.unsqueeze(0)  # Shape: (1, 1, clip_length, H, W)

        clip = clip-clip.min()
        clip = clip / clip.max()  # Normalize to [0, 1]
        return clip, keypoints
    
class ValidationClipDataset(Dataset):
    def __init__(self, videos, keypoints, clip_length=64, transform=None):
        """
        videos: list of tensors of shape (C, H, W, T)
        keypoints: list of tensors of shape (T, num_points, 2)
        clip_length: number of frames per clip
        transform: optional transform for the video clip
        """
        self.clip_length = clip_length
        self.transform = transform

        self.clips = []  # list of (video_index, start_frame)

        for vid_idx, video in enumerate(videos):
            T = video.shape[2]
            num_clips = T // clip_length

            for i in range(num_clips):
                start = i * clip_length
                self.clips.append((vid_idx, start))

        self.videos = videos
        self.keypoints = keypoints

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        vid_idx, start = self.clips[idx]
        video = self.videos[vid_idx]  # Shape: (C, H, W, T)
        
        end = start + self.clip_length
        keypts = self.keypoints[vid_idx][start:end]  # Shape: (T, N, 2)

        clip = video[:, :, start:end]  # Shape: (C, H, W, clip_length)

        clip = clip.permute(2,0,1)  # To (clip_length, C, H, W)

        if clip.shape[1] != 256 or clip.shape[2] != 256:
            clip, keypts = resize_or_crop_image_torch(clip, keypts, target_size=(256, 256))
        # clip, keypts = apply_transform_val(clip, keypts, version=0)
        
        clip = clip.unsqueeze(0)  # Shape: (1, clip_length, C, H, W)

        

        clip = clip - clip.min()
        clip = clip / clip.max()

        return clip, keypts
    

class RandomClipDataset_gaussian_map(Dataset):
    def __init__(self, videos, keypoints, clip_length=64, transform=None, sigma = 5):
        """
        videos: list of videos, where each video is a tensor of shape (T, C, H, W)
        clip_length: number of frames per clip (e.g., 64)
        transform: optional transform applied to each clip
        """
        self.videos = videos
        self.keypoints = keypoints
        self.clip_length = clip_length
        self.transform = transform
        self.sigma = sigma

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]  # Shape: (T, C, H, W)
        T = video.shape[2]

        if T < self.clip_length:
            raise ValueError(f"Video too short: {T} < {self.clip_length}")

        # Scegli inizio clip casuale
        start = random.randint(0, T - self.clip_length)
        end = start + self.clip_length

        clip = video[:, :, start:end]  # Shape: (clip_length, C, H, W)

        keypoints = self.keypoints[idx][start:end]

        clip = clip.permute(2,0,1)

        if clip.shape[1] != 256 or clip.shape[2] != 256:
            clip, keypoints = resize_or_crop_image_torch(clip, keypoints, target_size=(256, 256))

        
        if clip.max() > 1:
            clip = clip / 255.0

        if self.transform:
            clip, keypoints = apply_transform(clip, keypoints, version=self.transform)
        clip = clip.unsqueeze(0)  # Shape: (1, 1, clip_length, H, W)

        old_keyp = keypoints.clone()

        C, T, H, W = 3, 64, 256, 256
        
        # Create meshgrid (H, W, 2)
        y = torch.arange(H, dtype=torch.float32, device = keypoints.device).view(H, 1).expand(H, W)
        x = torch.arange(W, dtype=torch.float32, device = keypoints.device).view(1, W).expand(H, W)
        grid = torch.stack((x, y), dim=-1)  # shape: (H, W, 2)

        # Reshape for broadcasting
        grid = grid.view(1, 1, H, W, 2)           # (1, 1, H, W, 2)
        keypoints = torch.transpose(keypoints, 1,0)      # (C, T, 1, 1, 2)
        keypoints = keypoints.unsqueeze(2).unsqueeze(2)


        # Compute squared distance: (x - cx)^2 + (y - cy)^2
        squared_dist = ((grid - keypoints) ** 2).sum(dim=-1)  # (1, C, T, H, W)

        # Apply Gaussian
        gaussians = torch.exp(-squared_dist / (2 * self.sigma**2)) # (1, C, T, H, W)

        return clip, gaussians
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_path = r'c:\Users\vcxr10\Desktop\dataset_division_by_patient.txt'  # Path to the dataset division text file
    videos = []
    keypoints = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()	
        lines = [line.strip() for line in lines]
        for line in lines:
            if 'training' in line:
                training_flag = True
                test_flag = False
                val_flag = False
            elif 'test' in line:
                training_flag = False
                test_flag = True
                val_flag = False
            elif 'val' in line:
                training_flag = False
                test_flag = False
                val_flag = True
            elif training_flag:
                with h5py.File(line, 'r') as h5_file:
                    frames = h5_file['frames'][()]
                    annotations = h5_file['annotations'][()]
                    if frames.shape[2] > 64:
                        videos.append(torch.tensor(frames, dtype=torch.float32).to(device))
                        keypoints.append(torch.tensor(annotations, dtype=torch.float32).to(device))
    dataset = RandomClipDataset(videos, keypoints, clip_length=64)
    print(f"Number of clips: {len(dataset)}")
    print(f"Clip shape: {dataset[0][0].shape}")
    print(f"Keypoint shape: {dataset[0][1].shape}")
    print(dataset)


