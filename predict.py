import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.tasken_unet import UNet
import numpy as np
from utils.plot import visualize_image

def preprocess_images(images_array, model_type="EchoCoder"):
    """
    images_array: NumPy array con shape (N, 256, 256), valori tra 0-255 o normalizzati 0-1
    model_type: Specifica il modello per la corretta formattazione dell'input
    """

    # Assicuriamoci che il tipo di dato sia float32 e normalizziamo se necessario
    # if images_array.max() > 1:
    #     images_array = images_array.astype(np.float32)/256.0

    # Aggiungiamo le dimensioni richieste per PyTorch: (N, 1, 256, 256)
    images_tensor = torch.tensor(images_array).unsqueeze(1).unsqueeze(1)  # Shape diventa (N, 1, 256, 256)

    # Modifichiamo il formato in base al modello
    if model_type == "EchoCoder":
        images_tensor = images_tensor.repeat(1, 3, 1, 1)  # Shape diventa (N, 3, 256, 256)
    elif model_type == "EchoCoder Old":
        images_tensor = images_tensor.unsqueeze(1)  # Shape diventa (N, 1, 1, 256, 256)
    elif model_type == "EchoCoder 2D+t":
        images_tensor = images_tensor.repeat(1, 64, 1, 1)  # Shape diventa (N, 64, 256, 256)
    elif model_type == "U-Net":
        images_tensor = images_tensor.repeat(1, 1, 3, 1, 1)  # Shape diventa (N, 3, 256, 256)

    print(images_tensor.shape)
    return images_tensor

model = UNet(num_classes=2, depth=6, start_filts=8)
model_path = r'dl_mapse/Data/best_loss_weights_unet_light.pth'

# Load weights
model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
model.eval()  # Set to evaluation mode

test_path = r'data/dataset_256/test.npz'
data = np.load(test_path)
images = data['images']

input_tensor = preprocess_images(images, model_type="U-Net")
for im in input_tensor:
    visualize_image(im[0,1].numpy())
    im = im.unsqueeze(0)
    output = model(im)
    visualize_image(np.maximum(output[0,1].detach().numpy(), -200))
