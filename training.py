import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataloader.main import KeypointDataset
import torch.optim as optim
from models.improved_unet import ImprovedUNet
from losses.mse_considering_switched_points import UnorderedMSELoss

# Supponiamo che il tuo dataset sia definito come MyDataset
train_dataset = KeypointDataset(r'data/dataset/train.npz', filter=True)  # Assumendo che il dataset abbia un parametro "split"
val_dataset = KeypointDataset(r'data/dataset/val.npz', filter=True)

# Creiamo DataLoader per batch processing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Numero di classi del tuo problema
num_classes = 2  # Cambia in base al tuo dataset

# Definizione del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedUNet(in_channels=1, num_classes=num_classes).to(device)

# Definizione della loss function (usa BCEWithLogitsLoss per segmentazione binaria)
criterion = UnorderedMSELoss()

# Ottimizzatore (puoi provare Adam o SGD)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()  # Reset del gradiente

            images.requires_grad_(True)  # Ensure gradients flow

            outputs = model(images)  # Forward pass
            print(images.requires_grad)
            print(outputs.requires_grad)

            print("Pred requires_grad:", outputs.requires_grad)
            print("Target requires_grad:", masks.requires_grad)

            loss = criterion(outputs, masks)  # Calcolo della loss
            print(loss, loss.requires_grad)
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
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")


# Allenamento del modello
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)
