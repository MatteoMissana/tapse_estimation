import os
import subprocess

python_exec = r"C:/Users/vcxr10/anaconda3/python.exe"
script_path = r"d:/mmissana/tapse_estimation/2d/statystical_analysis.py"
base_dir = r"2d/results_no_sudden_movement_4_pixels"

# Loop through all subfolders inside base_dir
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path) or not "best" in folder:
        continue  # skip files
    
    excel_path = os.path.join(folder_path, "best_unet.xlsx")

    cmd = [
        python_exec,
        script_path,
        "--automatic_path", excel_path
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
