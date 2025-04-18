import h5py
import os
import numpy as np

# Path to one of your new files
path = r'D:\mmissana/data/RV_PATIENTS/RV_patients_converted/_11010/P42A0G2A.h5'

def stampa_contenuto_gruppo(gruppo, indent=0):
    for chiave in gruppo:
        oggetto = gruppo[chiave]
        print("  " * indent + f"- {chiave} ({type(oggetto)})")
        if isinstance(oggetto, h5py.Group):
            stampa_contenuto_gruppo(oggetto, indent + 1)
        elif isinstance(oggetto, h5py.Dataset):
            print("  " * (indent + 1) + f"  shape: {oggetto.shape}, dtype: {oggetto.dtype}")

# Sostituisci 'nome_file.h5' con il tuo file
with h5py.File(path, 'r') as f:
    print("Contenuto del file:")
    stampa_contenuto_gruppo(f)
    times = f['ecg']['ecg_times'][()]

print(1/(times[1] - times[0]))
print(1/(times[2] - times[1]))
   