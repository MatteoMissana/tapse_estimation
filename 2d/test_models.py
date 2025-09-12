import numpy as np
import os
import torch
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from dataloader.main import KeypointDataset
from dataloader.preprocessing import preprocess_images, apply_lut, resize_or_crop_image_np_nokeypoints
from utils.plot import save_image, visualize_image
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from models.models import Unet


#script to just predict landmarks from h5 file, to save images with predictions or to calcualte stistics on the ground truth

def compute_keypoint_distance_stats(predictions, ground_truths):
    if predictions.shape != ground_truths.shape:
        raise ValueError("Shape mismatch: predictions and ground_truths must have the same shape.")

    distances = np.linalg.norm(predictions - ground_truths, axis=-1)  # (N, 3)

    stats = {
        'global': {
            'mean': np.mean(distances),
            'max': np.max(distances),
            'min': np.min(distances),
            'std': np.std(distances)
        },
        'per_point': []
    }

    for i in range(3):
        point_dists = distances[:, i]
        stats['per_point'].append({
            'mean': np.mean(point_dists),
            'max': np.max(point_dists),
            'min': np.min(point_dists),
            'std': np.std(point_dists)
        })

    return stats, distances/2

def process_h5_file_batched(
    file_path,
    model,
    device,
    save_model_path,
    folder,
    batch_size=8,
    save_images=True,
    prediction_stats=False
):
    """
    Processes a .h5 file in batches: loads and preprocesses image data,
    performs inference, extracts keypoints, and optionally saves output images.

    Parameters:
    - file_path (str): Path to the .h5 file with image data.
    - model (torch.nn.Module): Trained PyTorch model.
    - device (torch.device): Device for model execution.
    - save_model_path (str): Base directory to store results.
    - folder (str): Subfolder for organizing outputs.
    - save_image (Callable): Function to save image with keypoints.
    - batch_size (int): Number of images to process per batch.
    - save_images (bool): Whether to save visualizations of predictions.
    - prediction_stats (bool): Whether I'm using it to create statistics on the distance from the ground truth (so I feed it with the annotated files) or just to predict/save image results

    Returns:
    - np.ndarray: Array of shape (N, 3, 2) with keypoint coordinates.
    """
    
    # Load and reorient images
    with h5py.File(file_path, 'r') as f:
        images = f['tissue']['data'][()]  # (H, W, N)
        if prediction_stats:
            annotations = f['annotations'][()]
    images = apply_lut(images.transpose(1, 0, 2)[:, ::-1, :])  # (H, W, N)
    images = resize_or_crop_image_np_nokeypoints(images.transpose(2, 0, 1))  # (N, H, W)
    images = images / 255.0 if images.max() > 1 else images  # Normalize

    model.eval()
    N = len(images)
    coordinates_array = np.zeros((N, 3, 2))  # (N images, 3 points, 2 coords)

    if save_images:
        file_name = os.path.basename(file_path).replace(".h5", "")
        save_path = os.path.join(save_model_path, folder, file_name)
        os.makedirs(save_path, exist_ok=True)

    # Process in batches
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = images[start:end]  # shape: (B, H, W)
        batch_pre = preprocess_images(batch, model_type='U-Net', device=device)  # (B, 1, H, W)
        batch_pre = batch_pre.to(device).float()
        with torch.no_grad():
            outputs = model(batch_pre.unsqueeze(1))  # (B, 3, H, W)

        for i in range(end - start):
            for j in range(3):  # 3 keypoints
                coordinates_array[start + i, j] = center_of_mass(outputs[i, j].detach(), thresh=0.8)

            if save_images:
                keypoints = [tuple(coordinates_array[start + i, k]) for k in range(3)]

                bold_flag = False
                if prediction_stats:  
                    # error calculation
                    dists = np.linalg.norm(coordinates_array[start + i] - annotations[start + i], axis=-1)  # shape (3,)
                    if np.any(dists > 40):
                        bold_flag = True
                        print(f"[WARNING] Error > 40  in image {start + i} of file {file_path}")

                save_image(batch_pre[i, 0].cpu().numpy(), points=keypoints, save_folder=save_path, bold=bold_flag)

    
    if prediction_stats:
        stats, distances = compute_keypoint_distance_stats(coordinates_array, annotations)
        return coordinates_array, stats, distances

    return coordinates_array, None, None

def main():
    model_checkpoint = r'2d/runs/best_unet/best_model.pth'
    test_path = r'D:\mmissana\data\RV_PATIENTS\RV_patients_annotated_renamed'
    save_model_path = r'D:\mmissana\tapse_estimation\2d\results\predictions_boxplots'

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Unet(depth=6, start_filts=16, num_residuals=0).to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)['model_state_dict'])

    stats_list = []
    all_distances = []

    keypoint_names = ['FW annular point', 'Septal annular point', 'Apex']

    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)
        if folder in ['100', '111', '140', '149', '160', '170', '190', '198', '199', '920']:  # test set
            for file in os.listdir(folder_path):
                if 'interpolated' in file:
                    file_path = os.path.join(folder_path, file)

                    coordinates_array, stats, distances = process_h5_file_batched(
                        file_path=file_path,
                        model=model,
                        device=device,
                        save_model_path=save_model_path,
                        folder=folder,
                        save_images=True,
                        batch_size=64,
                        prediction_stats=True  
                    )

                    # Append distances to full list with keypoint index labels
                    for i in range(3):
                        for d in distances[:, i]:
                            all_distances.append({'file': f"{folder}/{file}", 'kp': keypoint_names[i], 'distance': d})

                    # Flatten the stats dict for DataFrame
                    row = {
                        "file": f"{folder}/{file}",
                        "mean_global": stats['global']['mean'],
                        "max_global": stats['global']['max'],
                        "min_global": stats['global']['min'],
                        "std_global": stats['global']['std'],
                    }

                    for i, kp_stats in enumerate(stats['per_point']):
                        row.update({
                            f"mean_kp{i}": kp_stats['mean'],
                            f"max_kp{i}": kp_stats['max'],
                            f"min_kp{i}": kp_stats['min'],
                            f"std_kp{i}": kp_stats['std'],
                        })

                    stats_list.append(row)

    # Create DataFrame
    df = pd.DataFrame(stats_list)

    # Compute mean across files (for each stat column, not 'file')
    avg_row = {"file": "AVERAGE"}
    for col in df.columns:
        if col != "file":
            avg_row[col] = df[col].mean()
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save to Excel
    output_excel_path = os.path.join(save_model_path, "keypoint_stats.xlsx")
    df.to_excel(output_excel_path, index=False)
    print(f"Saved stats to {output_excel_path}")

    # Boxplot
    df_box = pd.DataFrame(all_distances)

    # Compute stats
    grouped = df_box.groupby('kp')['distance']
    medians = grouped.median()
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='kp', y='distance', data=df_box)
    # plt.yscale('symlog')
    plt.title('Landmark error distribution in the test set')
    plt.ylabel('Distance (mm)')
    plt.xlabel('')
    plt.grid(True)
    boxplot_path = os.path.join(save_model_path, "keypoint_distance_boxplot.pdf")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"Saved boxplot to {boxplot_path}")
                
    print(medians)
    print(q1)
    print(q3)


            
if __name__=='__main__':
    main()


# for im in images:
#     im = np.expand_dims(im, axis=0)
#     im = preprocess_images(im, model_type='monai_U-Net', device=device)
#     im = im.unsqueeze(0)
#     output = model(im)
#     coordinates_1 = center_of_mass(output[0, 0].detach(), thresh=0.8)
#     coordinates_2 = center_of_mass(output[0, 1].detach(), thresh=0.8)
#     coordinates_3 = center_of_mass(output[0, 2].detach(), thresh=0.8)