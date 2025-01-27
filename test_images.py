import h5py
from utils.plot import view_volume, visualize_image

file = r"D:\mmissana\data\4DRVQ_Jinyang\voxels\100001.h5"

def print_structure(name, obj):
    print(name, obj)


with h5py.File(file, 'r') as h5_file:
    h5_file.visititems(print_structure)
    print(h5_file['FrameInfo']['frameTimes'][:])
    print(h5_file['FrameInfo']['endDiastole'][()])
    print(h5_file['FrameInfo']['endSystole'][()])
    image_0 = h5_file['Input']['grid00'][:]

print(image_0.shape)
image_0 = image_0.transpose(2,1,0)
image_1 =image_0[161,::-1,:]

# Call the function with your volume
visualize_image(image_1)
