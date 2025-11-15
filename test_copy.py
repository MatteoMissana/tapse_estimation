import os
import subprocess
import sys


filter = ['none', 'avg', 'kalman', 'both']
reduction = ['mean', 'min', 'max']
area_method = ['spline']

folder = r'c:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\results\results_2_frames_method'
script = '2d/indices_prediction_copy.py'

h5_dir = r"C:\Users\User\Desktop\final_reviewed_dataset"
model_path = r"C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\model_weights\best_unet\best_model.pth"


for f in filter:
    for r in reduction:
        for spline in area_method:
            cmd = [sys.executable,
                   script,
                   '--h5_dir', h5_dir,
                   '--excel_path', os.path.join(folder, f'{f}_{spline}_{r}', 'best_unet.xlsx'),
                   '--model_path', model_path,
                   '--depth', '6',
                   '--filters', '16',
                   '--residuals', '0',
                   '--threshold', '0.875',
                   '--two_dimensional',
                   '--filter', f,
                   '--reduction', r,
                   '--area_method', spline]
            subprocess.run(cmd, check=True)
    