import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import wandb  # Import wandb
from tqdm import tqdm  # Import tqdm for progress bar
import h5py

from temporal_pipeline.dataloader.data_prep import RandomClipDataset, ValidationDataset
from temporal_pipeline.models.models import EncoderDecoder
from temporal_pipeline.losses.distances import CombinedLossPenalty, CombinedLandmarkLoss
from models.weights_initialization import initialize_weights

from temporal_pipeline.postprocessing.coordinates_calculation_from_masks import center_of_mass_3d_global_threshold, center_of_mass_3d
from dataloader.preprocessing import preprocess_images
from utils.plot import save_image, visualize_image
from temporal_pipeline.utils.save import get_experiment_path
from callbacks.early_stopping import EarlyStopping
from callbacks.lr_schedule import ReduceLROnPlateau

from monai.networks.nets import UNet


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net model for keypoint detection.')
    parser.add_argument('--epochs', type=int, default= 300, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--num_keypoints', type=int, default=3, help='')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--dataset_path', type=str, default='data/final_reviewed_dataset_for_3d/', help='Path of the dataset, divided into /train/, /val/ and /test/')
    parser.add_argument('--model', type=str, default='echocoder_2d+t', help='name of the model: supported "U-Net"')
    parser.add_argument('--save_images', action='store_true', help='If to save test images with predictions')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--wandb_project', type=str, default='rv_focused_training', help='tapse')
    parser.add_argument('--augm_version', type=str, default='8', help='augmentation version you want to use')
    parser.add_argument('--loss', type=str, default='ordered_distance', help='')
    parser.add_argument('--wandb_entity', type=str, default=None, help='master_thesis_NTNU_mmissana')
    parser.add_argument('--save_model_path', type=str, default=None, help='Path to save trained model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--from_scratch', action='store_true', help='Train model from scratch')

    return parser.parse_args()
args = parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

#set the tdevice: cuda then mps then cpu
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

def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                val_criterion,
                optimizer, 
                num_epochs, 
                patience, 
                checkpoint_path, 
                save_model_path=None,
                ):

    model.train()

    #set the early stopping callback and the lr scheduler
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience, path=checkpoint_path, delta=0.01)
    scheduler = ReduceLROnPlateau(optimizer, monitor='val_loss', mode='min', patience=7, factor=0.3, min_lr=0, initial_lr=args.initial_lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        
        # Use tqdm to create a progress bar for the training loop
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                # visualize_image(images[0, 0, 30].cpu().numpy(), points=[tuple(masks[0, 30, 0].tolist()), tuple(masks[0, 30, 1].tolist()), tuple(masks[0, 0, 2].tolist())])

                optimizer.zero_grad()
                # Forward pass
                outputs = model(images)

                if args.loss == 'ordered_distance':
                    # Compute center of mass for output masks
                    com_tensor = center_of_mass_3d_global_threshold(outputs, global_thresh=0.1, device=device, normalize=False).to(device)
                    

                    print(com_tensor.shape, masks.shape)
                    loss, loss_breakdown = criterion(com_tensor, masks)
                    print(loss_breakdown)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())  # Update progress bar with loss
                pbar.update(1)  # Move progress bar forward

        avg_loss = running_loss / len(train_loader)
        val_loss = validate(model, val_loader, val_criterion)
        scheduler(val_loss)

        # Log losses to wandb
        wandb.log({"train_loss": avg_loss, "val_loss": val_loss, "learning_rate": scheduler.currentlr})

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")

            # Save last model
            model_path = os.path.join(save_model_path, "last_model.pth")
            torch.save({'model_state_dict': model.state_dict()}, model_path)

            # load best model
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pth'), map_location=device)['model_state_dict'])
            # save best model
            model_path = os.path.join(save_model_path, "best_model.pth")
            torch.save({'model_state_dict': model.state_dict()}, model_path)

            break

    print("Training complete!")
    print(f"results saved in {save_model_path}")

def validate(model, val_loader, criterion):

    model.eval()

    # initialize val loss
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:

            # put images on device
            images, masks = images.to(device), masks.to(device)
            
            # compute outputs
            outputs = model(images)
            
            # Compute center of mass for output masks
            com_tensor = center_of_mass_3d_global_threshold(outputs, global_thresh=0.1, device=device, normalize=False).to(device)
            
            # calculate the loss: for the validation I use the distance loss, that's 
            # what I want to minimize
            loss, _ = criterion(com_tensor, masks)
            val_loss += loss.item()


    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def main():
    args = parse_args()

    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.model}_{args.loss}_augm_{args.augm_version}",  # Set a meaningful run name
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.initial_lr,
            "patience": args.patience,
            "model": "U-Net",
            "augmentation_version": args.augm_version,
            "loss_type": args.loss,
            "seed": args.seed
        }
    )

    train_path = args.dataset_path + "train"
    val_path = args.dataset_path + "val"
    # test_path = "data/final_reviewed_dataset_for_3d/test"

    # Load dataset
    train_dataset = RandomClipDataset(train_path, clip_length=32, transform=args.augm_version)
    val_dataset = ValidationDataset(val_path, clip_length=32)

    # set dataloader parameters
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    
    #load the model
    model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=3,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(device)

    # Calculate model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"model size: {size_all_mb:.3f} MB")

    # estabilish the loss
    criterion = CombinedLossPenalty(lambda_motion=.5, lambda_var=0, missing_penalty=20, reduction='mean')
    val_criterion = CombinedLossPenalty(lambda_motion=.5, lambda_var=0, missing_penalty=20, reduction='mean')

    # set the optimizer. TODO: write the code so that you can experiment with 
    # different optimizers
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=1e-5)

    # set the path where to save model and training. If not specified it 
    # automatically creates an increasing path
    save_model_path = args.save_model_path if args.save_model_path else get_experiment_path()
    
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    train_model(model, 
                train_loader, 
                val_loader, 
                criterion,
                val_criterion, 
                optimizer, 
                args.epochs, 
                args.patience, 
                args.checkpoint_path, 
                save_model_path)

   

    #Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main() # Run inference on test set
    