import subprocess

subprocess.run([
    "python",
    "d:/mmissana/tapse_estimation/2d/training.py",
    "--from_scratch",
    "--initial_lr", "5e-3",
    "--batch_size", "128",
    "--augm_version", "7",
    "--thresh", "0.875",
    "--model", "monai_U-Net",
    "--wandb_project", "monai_unet_reviewed_2",
    "--start_filts", "16",
    "--depth", "6",
    "--dropout", "0",
    "--n_residuals", "0",
])


subprocess.run([
    "python",
    "d:/mmissana/tapse_estimation/2d/training.py",
    "--from_scratch",
    "--initial_lr", "1e-2",
    "--batch_size", "128",
    "--augm_version", "7",
    "--thresh", "0.875",
    "--model", "monai_U-Net",
    "--wandb_project", "monai_unet_reviewed_2",
    "--start_filts", "16",
    "--depth", "6",
    "--dropout", "0",
    "--n_residuals", "0",
])