import numpy as np
import os
import torch


from dataloader.main import KeypointDataset
from dataloader.preprocessing import preprocess_images
from utils.plot import save_image, visualize_image
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from models.models import Unet

model_checkpoint = r'D:\mmissana\tapse_estimation/runs/exp0/best_model.pth'
test_path = r'D:\mmissana\data\RV_PATIENTS\dataset_256\test.npz'
save_model_path = r'D:\mmissana\test_results\monai_unet'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet(depth=6, start_filts=12, num_residuals=2).to(device)
model.load_state_dict(torch.load(model_checkpoint, map_location=device)['model_state_dict'])

data = np.load(test_path)
images = data['images']

os.makedirs(save_model_path, exist_ok=True)
model.eval()
for im in images:
    im = np.expand_dims(im, axis=0)
    im = preprocess_images(im, model_type='monai_U-Net', device=device)
    im = im.unsqueeze(0)
    output = model(im)
    coordinates_1 = center_of_mass(output[0, 0].detach(), thresh=0.8)
    coordinates_2 = center_of_mass(output[0, 1].detach(), thresh=0.8)
    coordinates_3 = center_of_mass(output[0, 2].detach(), thresh=0.8)

    # Save results
    save_image(im[0, 0, 0].cpu().numpy(), points=[tuple(coordinates_1.tolist()), tuple(coordinates_2.tolist()), tuple(coordinates_3.tolist())], save_folder=save_model_path)