import subprocess

# Comando 3
subprocess.run([
    "python.exe",
    "d:/mmissana/tapse_estimation/2d/training.py",
    "--from_scratch",
    "--initial_lr", "5e-3",
    "--batch_size", "128",
    "--augm_version", "7",
    "--thresh", "0.8",
    "--model", "monai_U-Net",
    "--wandb_project", "monai_unet",
    "--start_filts", "16",
    "--depth", "6",
    "--dropout", "0.25",
    "--n_residuals", "2",
])

subprocess.run([
    "python.exe",
    "d:/mmissana/tapse_estimation/2d/training.py",
    "--from_scratch",
    "--initial_lr", "1e-4",
    "--batch_size", "128",
    "--augm_version", "7",
    "--thresh", "0.8",
    "--model", "monai_U-Net",
    "--wandb_project", "monai_unet",
    "--start_filts", "12",
    "--depth", "6",
    "--dropout", "0.1",
    "--n_residuals", "2",
])

subprocess.run([
    "python.exe",
    "d:/mmissana/tapse_estimation/2d/training.py",
    "--from_scratch",
    "--initial_lr", "5e-3",
    "--batch_size", "128",
    "--augm_version", "7",
    "--thresh", "0.8",
    "--model", "monai_U-Net",
    "--wandb_project", "monai_unet",
    "--start_filts", "12",
    "--depth", "6",
    "--dropout", "0.25",
    "--n_residuals", "3",
])