import os
import pydicom

dicom_folder = "data/2d_focused_rv/RV focused TEE images"

folders = os.listdir(dicom_folder)
total_size = 0
dicom_files = 0

for f in folders:
    f_path = os.path.join(dicom_folder, f)
    for file in os.listdir(f_path):
        file_path = os.path.join(f_path, file)

        if os.path.isdir(file_path):
            continue

        try:
            ds = pydicom.dcmread(file_path)
            frames = getattr(ds, "NumberOfFrames", 1)  # Default to 1 if not present
            print(f"📄 {f}: {frames} frames")
        except:
            print(f"❌ Skipping non-DICOM file: {f}")
