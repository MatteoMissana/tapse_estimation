import numpy as np
import os
import torch
import h5py


from dataloader.main import KeypointDataset
from dataloader.preprocessing import preprocess_images, apply_lut, resize_or_crop_image_np_nokeypoints
from utils.plot import save_image, visualize_image
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from models.models import Unet

model_checkpoint = r'D:\mmissana\tapse_estimation/2d/runs/Best_monai_UNET/best_model.pth'
test_path = r'D:\mmissana\data\RV_PATIENTS\Test_set_converted'
save_model_path = r'D:\mmissana\test_results\monai_unet'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet(depth=6, start_filts=12, num_residuals=2).to(device)
model.load_state_dict(torch.load(model_checkpoint, map_location=device)['model_state_dict'])

for folder in os.listdir(test_path):
    folder_path = os.path.join(test_path, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        with h5py.File(file_path, 'r') as f:
            images = f['tissue']['data'][()]

        # Reorient and normalize images
        images = apply_lut(images.transpose(1, 0, 2)[:, ::-1, :])
        images = resize_or_crop_image_np_nokeypoints(images.transpose(2, 0, 1))
        images = images / 255.0 if images.max() > 1 else images


        model.eval()

        coordinates_array = np.zeros((len(images), 3, 2))

        save_path = os.path.join(save_model_path, folder, file.replace(".h5", ""))
        os.makedirs(save_path, exist_ok=True)
        for i, im in enumerate(images):
            im = preprocess_images(np.expand_dims(im, axis=0), model_type='U-Net', device=device)
            output = model(im.float().unsqueeze(0).to(device))

            coords = [center_of_mass(output[0, c].detach(), thresh=0.8) for c in range(3)]
            for j in range(3):
                coordinates_array[i, j] = coords[j]

            # Save results
            save_image(im[0, 0].cpu().numpy(), points=[tuple(coordinates_array[i,0].tolist()), tuple(coordinates_array[i,1].tolist()), tuple(coordinates_array[i,2].tolist())], save_folder=save_path)

# for im in images:
#     im = np.expand_dims(im, axis=0)
#     im = preprocess_images(im, model_type='monai_U-Net', device=device)
#     im = im.unsqueeze(0)
#     output = model(im)
#     coordinates_1 = center_of_mass(output[0, 0].detach(), thresh=0.8)
#     coordinates_2 = center_of_mass(output[0, 1].detach(), thresh=0.8)
#     coordinates_3 = center_of_mass(output[0, 2].detach(), thresh=0.8)

#     # Save results
#     save_image(im[0, 0, 0].cpu().numpy(), points=[tuple(coordinates_1.tolist()), tuple(coordinates_2.tolist()), tuple(coordinates_3.tolist())], save_folder=save_model_path)