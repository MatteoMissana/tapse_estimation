import os

path = r"D:\mmissana\data\RV_PATIENTS\RV_patients_annot"

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    print(folder_path)
    if not os.path.isdir(folder_path):
        continue    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        new_name = file_path.split('_interpolated')[0] + '.h5'
        print(new_name)
        os.rename(file_path, new_name)

    