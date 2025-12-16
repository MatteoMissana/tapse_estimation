import os
import subprocess
import sys

script_path = r"2d/statystical_analysis.py"
base_dir = r"C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\results\results_es_ed_method\results_2_frames_method"


# Loop through all subfolders inside base_dir
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue  # skip files
    
    excel_path = os.path.join(folder_path, "best_unet.xlsx")

    cmd = [
        sys.executable,
        script_path,
        "--automatic_path", excel_path
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
