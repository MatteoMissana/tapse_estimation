import numpy as np

def rename_npz_key(input_file, output_file, old_key='arr_0', new_key='video'):
    # Carica i dati dal file .npz
    data = np.load(input_file)
    
    # Verifica che la vecchia chiave esista
    if old_key not in data:
        raise KeyError(f"Key '{old_key}' not found in {input_file}")
    
    # Crea un nuovo dizionario con la chiave rinominata
    new_data = {new_key: data[old_key]}
    
    # Mantieni eventuali altre chiavi nel file originale
    for key in data.files:
        if key != old_key:
            new_data[key] = data[key]
    
    # Salva il nuovo file .npz
    np.savez(output_file, **new_data)
    print(f"Saved new .npz file with key '{new_key}' instead of '{old_key}'")

# Esempio di utilizzo
rename_npz_key(r'D:\mmissana\data\best_slices\199001\video_best_slice.npz', r'D:\mmissana\data\best_slices\199001\video_best_slice.npz')