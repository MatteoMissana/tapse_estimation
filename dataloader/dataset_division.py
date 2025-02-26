import os
import numpy as np

def dataset_division_from_txt(txt_path, data_separated_by_video_path = r'D:\mmissana\data\dataset_separated_by_video', save_path = r'D:\mmissana\data\dataset'):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    test = []
    val = []

    flag_val = False
    flag_test = False
    for line in lines:
        if line.strip() == 'Validation:':
            flag_val = True
            continue
        if line.strip() == 'Test:':
            flag_test = True
            flag_val = False
            continue
        if line.strip() == 'Training:':
            flag_val = False
            flag_test = False
            continue
        
        if flag_val:
            val.append(line.strip())
        if flag_test:
            test.append(line.strip())

    img_val_list = []
    img_test_list = []
    img_train_list = []
    keypoint_val_list = []
    keypoint_test_list = []
    keypoint_train_list = []

    for file in os.listdir(data_separated_by_video_path):
        if file.split('.')[0] in val:
            data = np.load(os.path.join(data_separated_by_video_path, file))
            for i in range(data['images'].shape[0]):
                img_val_list.append(data['images'][i])
                keypoint_val_list.append(data['keypoints'][i])
        elif file.split('.')[0] in test:
            data = np.load(os.path.join(data_separated_by_video_path, file))
            for i in range(data['images'].shape[0]):
                img_test_list.append(data['images'][i])
                keypoint_test_list.append(data['keypoints'][i])
        else:
            data = np.load(os.path.join(data_separated_by_video_path, file))
            for i in range(data['images'].shape[0]):
                img_train_list.append(data['images'][i])
                keypoint_train_list.append(data['keypoints'][i])
    
    np.savez_compressed(os.path.join(save_path, 'val.npz'), images=np.array(img_val_list), keypoints=np.array(keypoint_val_list))
    np.savez_compressed(os.path.join(save_path, 'test.npz'), images=np.array(img_test_list), keypoints=np.array(keypoint_test_list))
    np.savez_compressed(os.path.join(save_path, 'train.npz'), images=np.array(img_train_list), keypoints=np.array(keypoint_train_list))
    

if __name__ == "__main__":
    txt_path = r'D:\mmissana\data\dataset_division.txt'
    dataset_division_from_txt(txt_path)