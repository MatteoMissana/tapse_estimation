import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataloader.main import KeypointDataset
import torch.optim as optim
from models.improved_unet import ImprovedUNet
from losses.mse_considering_switched_points import UnorderedMSELoss
from models.tasken_unet import UNet
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from dataloader.preprocessing import preprocess_images
import numpy as np
from utils.plot import save_image

# Supponiamo che il tuo dataset sia definito come MyDataset
train_dataset = KeypointDataset(r'data/dataset_256/train.npz', filter=True)  # Assumendo che il dataset abbia un parametro "split"
val_dataset = KeypointDataset(r'data/dataset_256/val.npz', filter=True)

# Creiamo DataLoader per batch processing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Numero di classi del tuo problema
num_classes = 2  # Cambia in base al tuo dataset

# Definizione del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(num_classes=2, depth=6, start_filts=8).to(device)

model_path = r'dl_mapse/Data/best_loss_weights_unet_light.pth'

# Load weights
model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])

# Definizione della loss function (usa BCEWithLogitsLoss per segmentazione binaria)
criterion = UnorderedMSELoss()

# Ottimizzatore (puoi provare Adam o SGD)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()  # Reset del gradiente

            images.requires_grad_(True)  # Ensure gradients flow

            outputs = model(images)  # Forward pass

            # Applica center_of_mass e organizza i risultati in una lista di liste
            com_tensor = torch.stack([
                torch.stack([center_of_mass(mask, device=device, normalize=False) for mask in output])  # (num_masks, 2)
                for output in outputs
            ]).to(device)  # (batch_size, num_masks, 2)

            loss = criterion(com_tensor, masks)  # Calcolo della loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Aggiornamento pesi

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validazione ogni 5 epoche (opzionale)
        if (epoch + 1) % 5 == 0:
            validate(model, val_loader, criterion)

    print("Training completo!")


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            com_tensor = torch.stack([
                torch.stack([center_of_mass(mask, device=device, normalize=False) for mask in output])  # (num_masks, 2)
                for output in outputs
            ]).to(device)  # (batch_size, num_masks, 2)

            loss = criterion(com_tensor, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")


# Allenamento del modello
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)

model.eval()

test_path = r'data/dataset_256/test.npz'
data = np.load(test_path)
images = data['images']

# flip = True
# if flip:
#     images = images[:,:, ::-1]

save_folder = r'D:\mmissana\data\results\zero_shot_unet_light_test_set_cazzoculo'

for im in images:
    im = np.expand_dims(im, axis = 0)
    im = preprocess_images(im, model_type='U-Net', device=device)
    im = im.unsqueeze(0)
    output = model(im)
    coordinates_1 = center_of_mass(output[0, 0].detach())
    coordinates_2 = center_of_mass(output[0, 1].detach()) 
    
    # saving if --save_folder is specified
    if save_folder:
        save_image(im[0, 0, 0].cpu().numpy(), points=[tuple(coordinates_1.tolist()), tuple(coordinates_2.tolist())], save_folder=save_folder)
