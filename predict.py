import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.tasken_unet import UNet
import numpy as np
from utils.plot import visualize_image, save_image, VolumeViewer
from dataloader.preprocessing import preprocess_images, resize_or_crop_image_np
from postprocessing.coordinates_calculation_from_masks import center_of_mass
import argparse  # Added argparse for command-line argument parsing
import os
import h5py

def predict_points(model, im, device = 'cpu'):
    """
    Predict keypoints from the image using the model.
    
    Args:
        model (torch.nn.Module): The trained model.
        image (torch.Tensor): The input image tensor.
        
    Returns:
        torch.Tensor: Predicted keypoints.
    """

    model.eval()
    im = np.expand_dims(im, axis=0)
    im = preprocess_images(im, model_type='U-Net', device=device)
    
    im = im.float()

    if im.max() > 1:
        im = im / 255.0

    im = im.unsqueeze(0).to(device)

    output = model(im)
    coordinates_1 = center_of_mass(output[0, 0].detach())
    coordinates_2 = center_of_mass(output[0, 1].detach())
    coordinates_3 = center_of_mass(output[0, 2].detach())

    return coordinates_1, coordinates_2, coordinates_3

if __name__ == "__main__":
    LUT = [5.416839879697600921e-109, 5.416839879697600921e-109, 5.416839879697600921e-109,
        5.416839879697600921e-109, 5.416839879697600921e-109, 5.416839879697600921e-109,
        5.416839879697600921e-109, 5.416839879697600921e-109, 5.416839879697600921e-109,
        5.416839879697600921e-109, 5.416839879697600921e-109, 5.416839879697600921e-109,
        5.416839879697600921e-109, 5.416839879697600921e-109, 5.416839879697600921e-109,
        5.416839879697600921e-109, 5.416839879697600921e-109, 5.416839879697600921e-109,
        5.416839879697600921e-109, 5.416839879697600921e-109, 5.416839879697600921e-109,
        5.416839879697600921e-109, 1.472449741265426833e-108, 1.472449741265426833e-108,
        4.002533375001032603e-108, 4.002533375001032603e-108, 4.002533375001032603e-108,
        1.088001374106615945e-107, 2.957494364572485672e-107, 2.957494364572485672e-107,
        8.039303188987418261e-107, 8.039303188987418261e-107, 2.185309177209735204e-106,
        2.185309177209735204e-106, 5.940286225974011103e-106, 1.614737210391071792e-105,
        4.389310816742700680e-105, 4.389310816742700680e-105, 1.193138383261041334e-104,
        3.243286386055492711e-104, 8.816166447703253065e-104, 2.396482505146208087e-103,
        6.514314845958948177e-103, 6.514314845958948177e-103, 1.770774367063119509e-102,
        4.813463784288744541e-102, 1.308435113677780416e-101, 3.556695393228056346e-101,
        9.668100456775821155e-101, 2.628062178737031233e-100, 7.143813664521359541e-100,
        1.941889887016583139e-99, 1.434873331311163542e-98, 3.900390102643631095e-98,
        1.060235953991769413e-97, 2.882020127614766811e-97, 7.834142942148438469e-97,
        5.788692168657643875e-96, 1.573529673260525569e-95, 1.162689902905403913e-94,
        3.160518835200571108e-94, 2.335325097502397273e-93, 6.348071776085115550e-93,
        1.725584815468590525e-92, 4.690625847353124236e-92, 3.465929752518635897e-91,
        9.421373864986962783e-91, 6.961506001738774640e-90, 5.143895837989021708e-89,
        1.398255858389167097e-88, 1.033179097829598747e-87, 2.808471967173908365e-87,
        7.634218314105463291e-87, 2.075195691772212795e-86, 1.533373738276407791e-85,
        1.133018457275138174e-84, 3.079863483719709203e-84, 2.275728405825291715e-83,
        6.186071172062962685e-83, 4.570922692235090021e-82, 1.242505609359374486e-81,
        9.180943650792429284e-81, 6.783850767682641964e-80, 5.012625388918081340e-79,
        3.703857020163976607e-78, 1.006812723312220456e-77, 7.439395693671140224e-77,
        2.022237412882273316e-76, 1.494242568914349970e-75, 4.061772422389840816e-75,
        3.001266429012797238e-74, 8.158287996349656898e-74, 6.028204767626013778e-73,
        4.454274320382982031e-72, 3.291288283333605643e-71, 2.431951376330516621e-70,
        6.610729233975208078e-70, 4.884704916468365468e-69, 1.327800461182051828e-68,
        9.811192095860170043e-68, 2.666958518969771289e-67, 1.970630611018864524e-66,
        5.356729380537723324e-66, 3.958117389958326745e-65, 1.075927857603146641e-64,
        7.950091298231918329e-64, 2.161058871057420474e-63, 1.596818523133501619e-62,
        4.340602774780607539e-62, 1.179898164724503386e-61, 8.718333730174673753e-61,
        2.369888815317537995e-60, 1.751124140460958445e-59, 4.760048930390988699e-59,
        3.517226858031385466e-58, 9.560813854754816412e-58, 7.064538992421674337e-57,
        1.920340796954020926e-56, 1.418950587775845334e-55, 3.857107598232361988e-55,
        2.850038442295058349e-54, 7.747207708100381429e-54, 2.105909393422711263e-53,
        5.724455236522166123e-53, 1.149787570183009674e-51, 3.125446658616554427e-51,
        2.309410069473307261e-50, 6.277627426309632450e-50, 4.638574122118750361e-49,
        1.260895174611576965e-48, 9.316825180075900245e-48, 2.532575658592998838e-47,
        1.871334361612990863e-46, 5.086814190143600111e-46, 1.382739457781496258e-45,
        3.758675541580754053e-45, 1.021713942375242704e-44, 7.549501637270265095e-44,
        5.578369111675877908e-43, 4.121888230671500676e-42, 1.120445387637354474e-41,
        8.279033825040511196e-41, 2.250474720380540399e-40, 1.662888395811708180e-39,
        4.520199309090378025e-39, 3.340000627321637406e-38, 9.079063012290218145e-38,
        2.467945200574313964e-37, 6.708570592353872135e-37, 4.957000445053917868e-36,
        1.347452423345346453e-35, 9.956401546738817165e-35, 2.706430540134165694e-34,
        7.356840957233302509e-34, 1.999796708891053013e-33, 5.436011054390753364e-33,
        4.016699063530042531e-32, 1.091852007478217846e-31, 2.967961471294569291e-31,
        8.067755734986599734e-31, 2.193043381086032512e-30, 1.620452057023324203e-29,
        4.404845380495582738e-29, 1.197361115497291082e-28, 3.254764962359738827e-28,
        8.847368453087666257e-28, 2.404964089571001357e-27, 6.537370182777404428e-27,
        1.777041457375380675e-26, 4.830499502001877147e-26, 1.313065901867216908e-25,
        3.569283180614843283e-25, 2.637363365453269652e-24, 7.169096911355215233e-24,
        1.948762586059874693e-23, 5.297285925667413630e-23, 1.439951607189358438e-22,
        3.914194287683230482e-22, 1.063988320526752206e-21, 2.892220117380528479e-21,
        7.861869388979176538e-21, 7.861869388979176538e-21, 2.137077669778051444e-20,
        5.809179395763277227e-20, 5.809179395763277227e-20, 4.292435244404691073e-19,
        1.166804872470243210e-18, 3.171704482193336260e-18, 8.621586659188251145e-18,
        2.343590234815634879e-17, 6.370538748653406648e-17, 6.370538748653406648e-17,
        1.731691971795878072e-16, 4.707226819421149006e-16, 1.279556912566757742e-15,
        3.478196303909377189e-15, 3.478196303909377189e-15, 9.454717808730274267e-15,
        6.986144028826667324e-14, 1.899030836455719177e-13, 5.162101014420962350e-13,
        6.986144028826667324e-14, 1.899030836455719177e-13, 5.162101014420962350e-13,
        1.403204538417050386e-12, 1.403204538417050386e-12, 1.403204538417050386e-12,
        3.814305398390331470e-12, 1.036835705263767578e-11, 1.036835705263767578e-11,
        2.818411656716017816e-11, 7.661237191568304403e-11, 7.661237191568304403e-11,
        2.082540184135473035e-10, 1.538800624885437650e-9, 1.538800624885437650e-9,
        4.182893776247508078e-9, 1.137028414234803735e-8, 1.137028414234803735e-8,
        3.090763676856071573e-8, 8.401566738859123968e-8, 8.401566738859123968e-8,
        2.283782619682667372e-7, 2.283782619682667372e-7, 6.207964795233989027e-7,
        6.207964795233989027e-7, 1.687499789459803066e-6, 1.687499789459803066e-6,
        4.587100013217047569e-6, 4.587100013217047569e-6, 3.389433932906628784e-5,
        3.389433932906628784e-5, 9.213436668582564723e-5, 9.213436668582564723e-5,
        2.504471747386622835e-4, 2.504471747386622835e-4, 2.504471747386622835e-4,
        2.504471747386622835e-4, 2.504471747386622835e-4, 6.807860040810129361e-4,
        6.807860040810129361e-4, 1.850568223962662786e-3, 1.850568223962662786e-3,
        1.850568223962662786e-3, 5.030365975521435516e-3, 5.030365975521435516e-3,
        5.030365975521435516e-3, 1.367395242175857561e-2, 1.367395242175857561e-2,
        1.367395242175857561e-2, 3.716965639127988580e-2, 3.716965639127988580e-2,
        3.716965639127988580e-2, 2.746486762531433201e-1, 2.746486762531433201e-1,
        2.746486762531433201e-1
        ]
    
    LUT = np.log(np.array(LUT)).astype(int) + 249  # Simulate a tone curve, like photo filters

    folder = r'data/RV_PATIENTS/RV_patients_converted'
    save_folder = r'data/RV_PATIENTS/RV_patients_predicted'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    model = UNet(num_classes=3, depth=6, start_filts=8).to(device)
    checkpoint = torch.load(r'runs/U-Net_ordered_distance_augm_7_best_lr_0.0005/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


    # Loop through each subfolder inside the main folder
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        # Loop through each file in the current subfolder
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)

            # Open the original HDF5 file in read mode
            with h5py.File(file_path, 'r') as h5_file:

                # Extract image data from the 'tissue/data' path
                frames = h5_file['tissue']['data'][()]

                # Rearrange axes so the frame format matches expected input (H, W, C)
                frames = frames.transpose(1, 0, 2)

                # Flip the frames vertically (mirror image over horizontal axis)
                frames = frames[:, ::-1, :]

                # Convert frames to 8-bit integers
                frames = frames.astype(np.uint8)

                # Apply the LUT (Look-Up Table) transformation for contrast/styling
                # for i in range(frames.shape[2]):       # Loop over each color channel
                #     for j in range(frames.shape[0]):   # Loop over image height
                #         for k in range(frames.shape[1]):  # Loop over image width
                #             frames[j, k, i] = LUT[frames[j, k, i]]

                frames = np.take(LUT, frames)

                # Rearrange back to expected model input shape (N_frames, H, W)
                frames = frames.transpose(2, 0, 1)
                
                # Initialize keypoints array to store predictions (N_frames x 3 keypoints x 2 coordinates)
                keypoints = np.zeros((frames.shape[0], 3, 2))

                # Resize or crop frames and adjust keypoints accordingly
                frames, _ = resize_or_crop_image_np(frames, keypoints)

                # Predict keypoints for each frame using the model
                for i, frame in enumerate(frames):
                    c1, c2, c3 = predict_points(model, frame, device=device)

                    # Store predicted keypoints for the current frame
                    keypoints[i, 0] = c1.cpu().numpy()
                    keypoints[i, 1] = c2.cpu().numpy()
                    keypoints[i, 2] = c3.cpu().numpy()

                # Define path to save the new file with added keypoints
                new_file_path = os.path.join(save_folder, subfolder, file)

                os.makedirs(os.path.join(save_folder, subfolder), exist_ok=True)

                # Create the new HDF5 file and copy existing content + add keypoints
                with h5py.File(new_file_path, 'a') as h5_file:

                    # Copy all original datasets from the original file into the new file
                    with h5py.File(file_path, 'r') as original_file:
                        for item in original_file:
                            if item != 'tissue/data':
                                original_file.copy(item, h5_file)
                    
                    frames = frames.transpose(1, 2, 0)  # Rearrange frames to (N_frames, H, W) for saving
                    # Add the new keypoints dataset to the new file
                    h5_file.create_dataset('frames', data=frames)
                    h5_file.create_dataset('annotations', data=keypoints)

                    


                

                