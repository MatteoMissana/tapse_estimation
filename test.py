import os
import subprocess

base_dir = r"D:\mmissana\tapse_estimation\2d\results"
h5_dir = r"D:\mmissana\data\RV_PATIENTS\RV_patients_converted"
model_path = r"D:\mmissana\tapse_estimation\2d\runs\best_unet\best_model.pth"
script_path = r"d:/mmissana/tapse_estimation/2d/indices_prediction.py"

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path) or not "best" in folder:
        continue

    excel_path = os.path.join(folder_path, "best_unet.xlsx")

    cmd = [
        r"C:/Users/vcxr10/anaconda3/python.exe", script_path,
        "--h5_dir", h5_dir,
        "--model_path", model_path,
        "--depth", "6",
        "--filters", "16",
        "--residuals", "0",
        "--threshold", "0.875",
        "--two_dimensional",
        # "--no_sudden_movements",
        # "--threshold_sudden", "4",
        "--excel_path", excel_path
    ]

    if folder == "best_combination":
        cmd.append("--best_combination")
    else:
        if "nofilter" in folder:
            pass  # no --filter flag
        elif "filter" in folder:
            cmd.append("--filter")


        if "distance" in folder:
            cmd.extend(["--tapse", "distance"])
        elif "projection" in folder:
            cmd.extend(["--tapse", "projection"])

        if "max" in folder:
            cmd.extend(["--reduction", "max"])
        elif "mean" in folder:
            cmd.extend(["--reduction", "mean"])
        elif "min" in folder:
            cmd.extend(["--reduction", "min"])

        if "spline" in folder:
            cmd.extend(["--area_method", "spline"])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)