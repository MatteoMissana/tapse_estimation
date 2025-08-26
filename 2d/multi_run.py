import subprocess

subprocess.run([
    "python",
    "d:/mmissana/tapse_estimation/2d/training.py",
    "--from_scratch",
    "--initial_lr", "1e-4",
    "--batch_size", "16",
    "--augm_version", "7",
    "--model", "swinunetr",
    "--thresh", "0.8",
    "--wandb_project", "swinunetr_reviewed",
    "--start_filts", "12",
])

subprocess.run([
    "python",
    "d:/mmissana/tapse_estimation/2d/training.py",
    "--from_scratch",
    "--initial_lr", "1e-4",
    "--batch_size", "16",
    "--augm_version", "7",
    "--model", "swinunetr",
    "--thresh", "0.7",
    "--wandb_project", "swinunetr_reviewed",
    "--start_filts", "12",
])

subprocess.run([
    "python",
    "d:/mmissana/tapse_estimation/2d/training.py",
    "--from_scratch",
    "--initial_lr", "1e-4",
    "--batch_size", "16",
    "--augm_version", "7",
    "--model", "swinunetr",
    "--thresh", "0.9",
    "--wandb_project", "swinunetr_reviewed",
    "--start_filts", "12",
])
