import torch
from torch.utils.data import Dataset
import random
import os
import h5py
import numpy as np
from temporal_pipeline.augmentations.augm_3d import apply_transform
from utils.plot import visualize_image, VolumeViewer
from temporal_pipeline.dataloader.preprocessing import resize_or_crop_image_torch


class RandomClipDataset(Dataset):
    ''' this dataset clas should select sequences of clip_length frames, so as to 
    train the model to learn movement patterns in the data you feed it.

    dataset_path: path to the dataset of hdf5 files organized like this:
     -dataset
        -id1
            -hdf5 file with acquisitions and annotations in ["frames"] and 
            ["annotations"] datasets
        -id2
            -...
        id3
        etc.
    clip_length: length of the clips to feed to the model
    transform: version of the augmentation to apply: see temporal_pipeline/augmentations/augm_3d.py'''

    def __init__(self, dataset_path, clip_length=64, transform=None):
        
        # initialize
        self.clip_length = clip_length
        self.transform = transform

        # initialize the paths of the video files
        self.video_files = []

        # extract the paths of each sequence
        for subfolder in os.listdir(dataset_path):
            if not subfolder.startswith("."):
                for file in os.listdir(os.path.join(dataset_path, subfolder)):
                    if not file.startswith("."):
                        self.video_files.append(os.path.join(dataset_path, subfolder, file))
    

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # idx = idx % len(self.video_files)

        # load acquisition and annotations
        with h5py.File(self.video_files[idx], 'r') as h5_file:
            # extract the number of frames
            T = h5_file['annotations'].shape[0]
            
            # verify the acquisition is long enough
            if T < self.clip_length:
                raise ValueError(f"Acquisition too short: {T} < {self.clip_length}")

            # temporal cropping of the acquisition selection of a random index, accounting for 
            # the length of the clip
            start = random.randint(0, T - self.clip_length - 1)
            end = start + self.clip_length

            # clip the acquisitions and the annotations. I'm doing it like this so I don't load 
            # the entire acquisition in RAM but just the clip I need
            clip_acq = h5_file['frames'][:, :, start:end]
            clip_ann = h5_file['annotations'][start:end]

        # convert to tensors and permute to (N, H, W)
        clip_acq = torch.from_numpy(clip_acq).float()
        clip_ann = torch.from_numpy(clip_ann).float()
        clip_acq = clip_acq.permute(2, 0, 1)

        # Normalize to [0, 1]
        clip_acq = clip_acq-clip_acq.min()
        clip_acq = clip_acq / (clip_acq.max() + 1e-7)

        # pad or trim images
        clip_acq, clip_ann = resize_or_crop_image_torch(clip_acq, clip_ann)

        # apply the image augmentation if requested
        if self.transform:
            clip_acq, clip_ann = apply_transform(clip_acq, clip_ann, version= self.transform)

        #unsqueeze to pass it through the model
        clip_acq = clip_acq.unsqueeze(0)

        return clip_acq, clip_ann
    

class ValidationDataset(Dataset):
    '''
    Validation dataset that returns all possible non-overlapping clips of length clip_length from each video.
    Precomputes the mapping from dataset index to (video, start_frame) for efficient access.
    dataset_path: path to the dataset of hdf5 files organized like this:
     -dataset
        -id1
            -hdf5 file with acquisitions and annotations in ["frames"] and 
            ["annotations"] datasets
        -id2
            -...
        id3
        etc.
    clip_length: length of the clips to feed to the model
    transform: to be added, apply augmentations
    '''
    def __init__(self, dataset_path, clip_length=64):
        self.clip_length = clip_length

        # list of paths of the files
        self.video_files = []

        self.clip_indices = []  # List of (video_idx, start_frame)

        # Gather all video files
        for subfolder in os.listdir(dataset_path):
            if not subfolder.startswith('.'):
                for file in os.listdir(os.path.join(dataset_path, subfolder)):
                    if not file.startswith('.'):
                        self.video_files.append(os.path.join(dataset_path, subfolder, file))

        # Precompute all possible (video_idx, start_frame) pairs: I have to track the idx of the
        # video and the number of clips i can obtain from that video
        for vid_idx, video_path in enumerate(self.video_files):
            with h5py.File(video_path, 'r') as h5_file:
                # extract numberof frame in the acquisition
                T = h5_file['annotations'].shape[0]
            # number of popossible clips
            n_clips = T // self.clip_length
            # track the video they belong to and the starting frame of the clip
            for j in range(n_clips):
                start = j * self.clip_length
                self.clip_indices.append((vid_idx, start))

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, idx):
        vid_idx, start = self.clip_indices[idx]
        video_path = self.video_files[vid_idx]
        with h5py.File(video_path, 'r') as h5_file:
            clip_acq = h5_file['frames'][:, :, start:start+self.clip_length]
            clip_ann = h5_file['annotations'][start:start+self.clip_length]

        # convert to tensors and permute to (N, H, W)
        clip_acq = torch.from_numpy(clip_acq).float()
        clip_ann = torch.from_numpy(clip_ann).float()
        clip_acq = clip_acq.permute(2, 0, 1)

        # Normalize to [0, 1]
        clip_acq = clip_acq-clip_acq.min()
        clip_acq = clip_acq / (clip_acq.max() + 1e-7)

        # pad or trim images
        clip_acq, clip_ann = resize_or_crop_image_torch(clip_acq, clip_ann)

        #unsqueeze to pass it through the model
        clip_acq = clip_acq.unsqueeze(0)

        return clip_acq, clip_ann

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() else "cpu")
    data_path = 'data/final_reviewed_dataset/train' 
    
    dataset = RandomClipDataset(data_path, clip_length=64)
    print(f"Number of clips: {len(dataset)}")
    print(f"Clip shape: {dataset[0][0].shape}")
    print(f"Keypoint shape: {dataset[0][1].shape}")
    print(dataset)


