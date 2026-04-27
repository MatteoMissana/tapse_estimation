import os
import h5py
import numpy as np
import pandas as pd
from pydicom import dcmread

'''this file uses a folder of dicom files to create the excel file that needs to be give as input to 
2d/indices_prediction.py. It extracts both the path and the patient id and writes it in the specific field in the excel 
file'''

dcm_path = r"C:\Users\User\Desktop\RV_followup_Jinyang\Postop RV images test data N = 10 x 3"
excel_path = r'c:\Users\user\Desktop\best_unet.xlsx'

# create an excel file with empty fields for each index
df = pd.DataFrame({'path': [], 'id': [],	'tapsefw': [], 'tapsesep': [],
                   'rvfac': [], 'rvad': [], 'rvas': [], 'rvldfw': [],
                    'rvldsep': [], 'rvlsfw': [], 'rvlssep': [], 'tadd': [], 
                    'tasd': [],	'rvldmid': [], 'rvlsmid': [], 'rvlsffw': [], 
                    'rvlsfglobal': [], 'rvlsfsep': [], 'rvlsfmid': []}
)

line = 1

for folder in os.listdir(dcm_path):
    folder_path = os.path.join(dcm_path, folder)
    for file in os.listdir(folder_path): # for each file (each one is in a separate folder)

        dcm = os.path.join(folder, file)
        # append the path of the file in the path column 
        df.loc[line, "path"] = dcm

        #read the dicom file to get the patient id
        ds = dcmread(os.path.join(dcm_path, dcm))

        # write the patient id on the excel
        df.loc[line, "id"] = ds.PatientID

        #increment the line of the excel
        line = line+1

df.to_excel(excel_path, index=False)

# select heartbeats of the patients that are NOT excluded from the study
heartbeat_col = df.loc[~df['id'].isin([100, 160, 170, 920]), 'Heartbeats']
print(heartbeat_col)
print ("hello")