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

from dataloader.dataset_creation import RandomClipDataset, ValidationClipDataset
from losses.distances import GaussianKeypointLoss,OrderedDistanceLoss
from models.tasken_unet_modified_for_sequence import UNet
from models.weights_initialization import initialize_weights
from models.models import EncoderDecoder_3d
from dataloader.preprocessing import preprocess_images
from utils.plot import save_image, visualize_image
from utils.save import get_experiment_path
from callbacks.early_stopping import EarlyStopping
from callbacks.lr_schedule import ReduceLROnPlateau
from postprocessing.coordinates_calculation_from_masks import center_of_mass


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net model for keypoint detection.')
    parser.add_argument('--epochs', type=int, default= 300, help='Number of training epochs')
    parser.add_argument('--clip_length', type=int, default= 5, help='Length of the sequence of frames for the prediction')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num_keypoints', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--model', type=str, default='Unet', help='name of the model: supported "U-Net"')
    parser.add_argument('--save_images', action='store_true', help='If to save test images with predictions')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--wandb_project', type=str, default='rv_focused_training', help='tapse')
    parser.add_argument('--augm_version', type=str, default='0', help='augmentation version you want to use')
    parser.add_argument('--loss', type=str, default='ordered_distance', help='select the type of loss: "MSE" or "distance"')
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(args.seed)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, checkpoint_path, save_model_path=None):
    model.train()
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience, path=checkpoint_path, delta=0.01)
    scheduler = ReduceLROnPlateau(optimizer, monitor='val_loss', mode='min', patience=3, factor=0.3, min_lr=0, initial_lr=args.initial_lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        
        # Use tqdm to create a progress bar for the training loop
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, masks in train_loader:
                
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                
                # print(masks[0, 0, 0])
                # print(masks[0, 1, 0])
                # print(masks[0, 2, 0])
                # print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
                # for i in range(5):
                #     visualize_image(images[0, 0, i].cpu().numpy(), points=[tuple(masks[0, i, 0].tolist()), tuple(masks[0, i, 1].tolist()), tuple(masks[0, i, 2].tolist())])

                # print(images.max(), images.min())

                # print(images.shape)
                outputs = model(images)  # Forward pass

                # visualize_image(masks[0, 0, 0].cpu().numpy())
                # visualize_image(outputs[0, 0, 0].cpu().detach().numpy())

                if args.loss == 'ordered_distance' or args.loss == 'gaussian':
                    # Compute center of mass for output masks
                    com_tensor = torch.stack([
                    torch.stack([center_of_mass(mask, device=device, normalize=False) for mask in output])
                    for output in outputs
                    ]).to(device)
                    
                    loss = criterion(com_tensor, masks)

                elif args.loss == 'MSE':
                    # print(outputs.shape, masks.shape)
                    loss = F.mse_loss(outputs, masks)


                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())  # Update progress bar with loss
                pbar.update(1)  # Move progress bar forward

        avg_loss = running_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion)
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
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Compute center of mass for output masks
            com_tensor = torch.stack([
                    torch.stack([center_of_mass(mask, device=device, normalize=False) for mask in output])
                    for output in outputs
                ]).to(device)

            # com_tensor = com_tensor.permute(0, 2, 1, 3)  # Rearrange dimensions to match masks


            loss = criterion(com_tensor, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

class Tester:
    """
    Evaluates the model using two different criteria and returns two metrics.

    Args:
        criterion1 (callable): First loss function.
        criterion2 (callable): Second loss function.
    """
    
    def __init__(self, criterion1, criterion2):
        self.criterion1 = criterion1
        self.criterion2 = criterion2
    
    def __call__(self, model, test_loader, device):
        """
        Runs evaluation on the validation set and computes two metrics.

        Args:
            model (torch.nn.Module): The model to evaluate.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            device (torch.device): Device to run the evaluation on.

        Returns:
            tuple: (metric1, metric2) computed using criterion1 and criterion2.
        """
        model.eval()
        val_loss1 = 0.0
        val_loss2 = 0.0

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                # Compute center of mass for output masks
                com_tensor = torch.stack([
                    torch.stack([center_of_mass(mask, device=device, normalize=False) for mask in output])
                    for output in outputs
                ]).to(device)
                
                loss1 = self.criterion1(com_tensor, masks)
                loss2 = self.criterion2(com_tensor, masks)
                
                val_loss1 += loss1.item()
                val_loss2 += loss2.item()

        avg_val_loss1 = val_loss1 / len(test_loader)
        avg_val_loss2 = val_loss2 / len(test_loader)

        return avg_val_loss1, avg_val_loss2

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
            "seed": args.seed,
            "clip_length": args.clip_length,
        }
    )

    # Load dataset
    txt_path = r'c:\Users\vcxr10\Desktop\dataset_division_by_patient.txt'  # Path to the dataset division text file
    videos = []
    keypoints = []
    videos_val = []
    keypoints_val = []
    videos_test = []
    keypoints_test = []
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
            elif val_flag:
                with h5py.File(line, 'r') as h5_file:
                    frames = h5_file['frames'][()]
                    annotations = h5_file['annotations'][()]
                    if frames.shape[2] > 64:
                        videos_val.append(torch.tensor(frames, dtype=torch.float32).to(device))
                        keypoints_val.append(torch.tensor(annotations, dtype=torch.float32).to(device))
            elif test_flag:
                with h5py.File(line, 'r') as h5_file:
                    frames = h5_file['frames'][()]
                    annotations = h5_file['annotations'][()]
                    if frames.shape[2] > 64:
                        videos_test.append(torch.tensor(frames, dtype=torch.float32).to(device))
                        keypoints_test.append(torch.tensor(annotations, dtype=torch.float32).to(device))


    print(f"Number of training videos: {len(videos)}")
    print(f"Number of validation videos: {len(videos_val)}")
    print(f"Number of test videos: {len(videos_test)}")


    train_dataset = ValidationClipDataset(videos, keypoints, clip_length=args.clip_length, transform = args.augm_version)
    val_dataset = ValidationClipDataset(videos_val, keypoints_val, clip_length=args.clip_length)
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, generator=g)

    if args.model == "Unet":
        model = UNet(num_classes= 3, in_channels= args.clip_length).to(device)
        if args.from_scratch:
            initialize_weights(model)
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])


    # Define loss function and optimizer
    if args.loss == 'MSE':
        criterion = OrderedDistanceLoss() 
    elif args.loss == 'distance':
        criterion = UnorderedDistanceLoss()
    elif args.loss == 'ordered_distance':
        criterion = OrderedDistanceLoss()
    elif args.loss == 'gaussian':
        criterion = GaussianKeypointLoss(sigma = 20)
    else:
        criterion = OrderedDistanceLoss()



    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=1e-5)

    # Train the model
    save_model_path = args.save_model_path if args.save_model_path else get_experiment_path()
    
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, args.patience, args.checkpoint_path, save_model_path)

    # Run inference on test set
    model.eval()
    test_dataset = ValidationClipDataset(videos_test, keypoints_test, clip_length=args.clip_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, generator=g)

    metric1 = OrderedDistanceLoss()
    metric2 = OrderedDistanceLoss()
    tester = Tester(metric2, metric1)
    test_distance, test_MSE = tester(model, test_loader, device=device)

        # Log final test results to wandb
    wandb.summary["test_distance"] = test_distance
    wandb.summary["test_MSE"] = test_MSE

    if args.save_images:

        os.makedirs(save_model_path, exist_ok=True)

        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            # Compute center of mass for output masks
            com_tensor = torch.stack([
                torch.stack([
                    torch.stack([center_of_mass(mask, device=device, normalize=False) for mask in frame])
                for frame in output])
            for output in outputs]).to(device)

            for i, im in enumerate(images[0, 0]):
                im = im.cpu().numpy()
                coordinates_1 = com_tensor[0,i,0].cpu().detach().numpy()
                coordinates_2 = com_tensor[0,i,1].cpu().detach().numpy()
                coordinates_3 = com_tensor[0,i,2].cpu().detach().numpy()

                # Save results
                save_image(im, points=[tuple(coordinates_1.tolist()), tuple(coordinates_2.tolist()), tuple(coordinates_3.tolist())], save_folder=save_model_path)

    #Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()