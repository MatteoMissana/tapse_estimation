import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import random
import os
import wandb  # Import wandb

from dataloader.main import KeypointDataset
from losses.mse_considering_switched_points import UnorderedMSELoss
from models.tasken_unet import UNet
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from dataloader.preprocessing import preprocess_images
from utils.plot import save_image
from callbacks.early_stopping import EarlyStopping


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net model for keypoint detection.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save test images')
    parser.add_argument('--train_data', type=str, default='D:/mmissana/data/dataset_256/train.npz', help='Path to the training dataset')
    parser.add_argument('--val_data', type=str, default='D:/mmissana/data/dataset_256/val.npz', help='Path to the validation dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--model_path', type=str, default='dl_mapse/Data/best_loss_weights_unet_light.pth', help='Path to the pre-trained model weights')
    parser.add_argument('--wandb_project', type=str, default='unet-training', help='tapse')
    parser.add_argument('--wandb_entity', type=str, default=None, help='master_thesis_NTNU_mmissana')
    return parser.parse_args()

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(seed)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, checkpoint_path):
    model.train()
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience, path=checkpoint_path)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)  # Forward pass

            # Compute center of mass for output masks
            com_tensor = torch.stack([
                torch.stack([center_of_mass(mask, device=device, normalize=False) for mask in output])
                for output in outputs
            ]).to(device)

            loss = criterion(com_tensor, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion)

        # Log losses to wandb
        wandb.log({"train_loss": avg_loss, "val_loss": val_loss})

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pth'), map_location=device)['model_state_dict'])
            break

    print("Training complete!")

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            com_tensor = torch.stack([
                torch.stack([center_of_mass(mask, device=device, normalize=False) for mask in output])
                for output in outputs
            ]).to(device)

            loss = criterion(com_tensor, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

def main():
    args = parse_args()

    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.initial_lr,
            "patience": args.patience,
            "model": "U-Net"
        }
    )

    # Load dataset
    train_dataset = KeypointDataset(args.train_data, filter=True, transform='0')
    val_dataset = KeypointDataset(args.val_data, filter=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, generator=g)

    # Define model
    model = UNet(num_classes=2, depth=6, start_filts=8).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])

    # Define loss function and optimizer
    criterion = UnorderedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=1e-5)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, args.patience, args.checkpoint_path)

    # Run inference on test set
    model.eval()
    test_path = 'D:/mmissana/data/dataset_256/test.npz'
    data = np.load(test_path)
    images = data['images']

    os.makedirs(args.save_path, exist_ok=True)

    for im in images:
        im = np.expand_dims(im, axis=0)
        im = preprocess_images(im, model_type='U-Net', device=device)
        im = im.unsqueeze(0)
        output = model(im)
        coordinates_1 = center_of_mass(output[0, 0].detach())
        coordinates_2 = center_of_mass(output[0, 1].detach())

        # Save results
        save_image(im[0, 0, 0].cpu().numpy(), points=[tuple(coordinates_1.tolist()), tuple(coordinates_2.tolist())], save_folder=args.save_path)

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
