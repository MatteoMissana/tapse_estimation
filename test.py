import os
import h5py
import numpy as np
import pandas as pd


dataset_path = r'c:\Users\vcxr10\Desktop\NewRVData\RV_patients_out2'
excel_path = r'c:\Users\vcxr10\Desktop\best_unet.xlsx'

df = pd.read_excel(excel_path)

df.to_excel("file.xlsx", index=False)
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        df.loc[len(df), "path"] = os.path.join(folder, file).replace('.h5', '')

df.to_excel(excel_path, index=False)
    