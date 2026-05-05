import torch
from torch.utils.data import DataLoader
import argparse

from monai.networks.nets import UNet
from temporal_pipeline.utils.plot import save_image
from temporal_pipeline.dataloader.data_prep import ValidationDataset
from temporal_pipeline.postprocessing.coordinates_calculation_from_masks import center_of_mass_3d, argmax_3d
from temporal_pipeline.utils.plot import visualize_image
from temporal_pipeline.models.models import UNet3D

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='test the trained model and save the images with predictions')
    parser.add_argument('--heatmap_method', action='store_true', help='if the  model was trained with the heatmap method')
    parser.add_argument('--unet_initial_channels', type=int, default=16, help='number of filters in the first layer of the UNet')
    parser.add_argument('--unet_res_units', type=int, default=2, help='number of residual units the UNet')
    parser.add_argument('--window_len', type=int, default=32, help='number of frames the model receives in input')
    return parser.parse_args()
args = parse_args()


# path to the test set
test_path = "data/final_reviewed_dataset_for_3d/test"

# load the test_set in 3d images of 32 frames 
test_dataset = ValidationDataset(test_path, clip_length=args.window_len, return_heatmaps=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#path to the modelweights
model_checkpoint = "/Users/mmissana/Desktop/other_model_5_mag/best.pt"

# path where to save the predictions
img_path = "/Users/mmissana/Desktop/other_model_5_mag"

#set the tdevice: cuda then mps then cpu
device = torch.device("cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() else "cpu")
print('Using device:', device)

#load the model
model = UNet3D(
            device=device,
            initial_channels=args.unet_initial_channels,
            num_res_units=args.unet_res_units,
        )

model.load_state_dict(torch.load(model_checkpoint, 
map_location=device)['model_state_dict'])
model.eval()

for images, masks in test_loader:
    images, masks = images.to(device), masks.to(device)

    outputs = model(images)

    mask = outputs[0, 0, 0].detach().cpu().numpy()
    visualize_image(mask)


    if args.heatmap_method:
        com_tensor = argmax_3d(outputs, device=device)
    else:
        # Compute center of mass for output masks
        com_tensor = center_of_mass_3d(outputs, device=device, normalize=False).to(device)

    for i, im in enumerate(images[0, 0]):
        im = im.cpu().numpy()
        coordinates_1 = com_tensor[0,i,0].cpu().detach().numpy()
        coordinates_2 = com_tensor[0,i,1].cpu().detach().numpy()
        coordinates_3 = com_tensor[0,i,2].cpu().detach().numpy()

        # Save results TODO: add the ground truth coordinates too
        save_image(im, points=[tuple(coordinates_1.tolist()), tuple(coordinates_2.tolist()), tuple(coordinates_3.tolist())], save_folder=img_path)