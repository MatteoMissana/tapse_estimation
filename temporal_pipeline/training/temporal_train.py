import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import os
import wandb  # Import wandb
import h5py

# TODO:maybe u have to set numpy seed if a run differs from the same one
from temporal_pipeline.dataloader.data_prep import RandomClipDataset, ValidationDataset
from temporal_pipeline.models.models import UNet3D, EncoderDecoder_3d
from temporal_pipeline.losses.distances import CombinedLossPenalty, CombinedLandmarkLoss
from temporal_pipeline.postprocessing.coordinates_calculation_from_masks import center_of_mass_3d_global_threshold, center_of_mass_3d
from temporal_pipeline.training.trainer import Trainer
from temporal_pipeline.utils.save import get_experiment_path

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train model for temporal window keypoint detection.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--stop_patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=10, help='Reduce on plateau patience')
    parser.add_argument('--dataset_path', type=str, default='data/final_reviewed_dataset_for_3d/', help='Path of the dataset, divided into /train/, /val/ and /test/')
    parser.add_argument('--model', type=str, default='3D_UNet', help='name of the model: supported "3D_UNet", "echocoder"')
    parser.add_argument('--save_images', action='store_true', help='If to save test images with predictions')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--wandb_project', type=str, default='rv_focused_training', help='tapse')
    parser.add_argument('--augm_version', type=str, default='8', help='augmentation version you want to use')
    parser.add_argument('--window_len', type=int, default=32, help='number of frames the model receives in input')
    parser.add_argument('--loss', type=str, default='ordered_distance', help='')
    parser.add_argument('--wandb_entity', type=str, default=None, help='master_thesis_NTNU_mmissana')
    parser.add_argument('--save_model_path', type=str, default=None, help='Path to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--reduce_factor', type=float, default=0.3, help='Factor by which the learning rate will be reduced by ReduceLROnPlateau. new_lr = lr * factor')
    parser.add_argument('--from_scratch', action='store_true', help='Train model from scratch')
    parser.add_argument('--wandb', action='store_true', help='if to log results on wandb')
    parser.add_argument('--smooth_annotations', action='store_true', help='Apply moving average smoothing to annotations')
    parser.add_argument('--smooth_window', type=int, default=3, help='Window size for moving average smoothing')

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    print('Using device:', device)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(args.seed)

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.model}_{args.loss}_augm_{args.augm_version}",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.initial_lr,
                "early stopping patience": args.stop_patience,
                "reduce on plateau patience": args.lr_patience,
                "model": args.model,
                "augmentation_version": args.augm_version,
                "loss_type": args.loss,
                "seed": args.seed,
                "smooth_annotations": args.smooth_annotations,
                "smooth_window": args.smooth_window,
            }
        )

    train_path = os.path.join(args.dataset_path, "train")
    val_path = os.path.join(args.dataset_path, "val")

    train_dataset = RandomClipDataset(train_path, clip_length=args.window_len, transform=args.augm_version, 
                                       smooth_annotations=args.smooth_annotations, smooth_window=args.smooth_window)
    val_dataset = ValidationDataset(val_path, clip_length=args.window_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # select the right model
    if args.model == "3D_UNet":
        model = UNet3D(device=device)
    elif args.model == "echocoder":
        model = EncoderDecoder_3d().to(device)
    else:
        raise Exception("Insert a valid model type. Accepted: '3D_UNet', 'echocoder'")
    

    train_loss = CombinedLandmarkLoss(lambda_motion=.5, lambda_var=0, reduction='mean')
    val_loss = CombinedLandmarkLoss(lambda_motion=0, lambda_var=0, reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=1e-5)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=args.reduce_factor,
        patience=args.lr_patience,
    )

    save_model_path = args.save_model_path if args.save_model_path else get_experiment_path()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loss_fn=train_loss,
        val_loss_fn=val_loss,
        device=device,
        scheduler=lr_scheduler,
        checkpoint_dir=save_model_path,
        model_type=args.model,
        wandb=args.wandb,
    )

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.stop_patience,
    )

    if args.wandb:
        wandb.finish()

    return history


if __name__ == "__main__":
    main()
