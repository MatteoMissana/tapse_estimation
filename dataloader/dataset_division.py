import os
import numpy as np

# Function to divide dataset based on a text file containing train/val/test splits
'''
The txtfile should be something like this:
Validation:
400001 
660001
730001  
135001 
Test:
106001 
157001 
184001 
661001 
800001 
920001 
112001 
Training:
other
'''
def dataset_division_from_txt(txt_path, data_separated_by_video_path = r'D:\mmissana\data\dataset_separated_by_video_256', save_path = r'D:\mmissana\data\dataset_256'):
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Read the dataset division file
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Lists to store video names for validation and test sets
    test = []
    val = []

    # Flags to track which section is being read
    flag_val = False
    flag_test = False
    
    for line in lines:
        line = line.strip()  # Remove leading/trailing spaces and newline characters
        
        if line == 'Validation:':
            flag_val = True
            continue
        if line == 'Test:':
            flag_test = True
            flag_val = False
            continue
        if line == 'Training:':
            flag_val = False
            flag_test = False
            continue
        
        # Add video names to respective lists
        if flag_val:
            val.append(line)
        if flag_test:
            test.append(line)

    # Lists to store images and keypoints for each dataset split
    img_val_list = []
    img_test_list = []
    img_train_list = []
    keypoint_val_list = []
    keypoint_test_list = []
    keypoint_train_list = []

    # Process each file in the dataset folder
    for file in os.listdir(data_separated_by_video_path):
        file_name = file.split('.')[0]  # Extract video name without extension
        file_path = os.path.join(data_separated_by_video_path, file)
        
        # Load the .npz file containing images and keypoints
        data = np.load(file_path)
        
        if file_name in val:  # If the file belongs to the validation set
            for i in range(data['images'].shape[0]):
                img_val_list.append(data['images'][i])
                keypoint_val_list.append(data['keypoints'][i])
        elif file_name in test:  # If the file belongs to the test set
            for i in range(data['images'].shape[0]):
                img_test_list.append(data['images'][i])
                keypoint_test_list.append(data['keypoints'][i])
        else:  # If the file belongs to the training set (default case)
            for i in range(data['images'].shape[0]):
                img_train_list.append(data['images'][i])
                keypoint_train_list.append(data['keypoints'][i])
    
    # Save the divided datasets into compressed .npz files
    np.savez_compressed(os.path.join(save_path, 'val.npz'), images=np.array(img_val_list), keypoints=np.array(keypoint_val_list))
    np.savez_compressed(os.path.join(save_path, 'test.npz'), images=np.array(img_test_list), keypoints=np.array(keypoint_test_list))
    np.savez_compressed(os.path.join(save_path, 'train.npz'), images=np.array(img_train_list), keypoints=np.array(keypoint_train_list))
    
# Main execution
if __name__ == "__main__":
    txt_path = r'D:\mmissana\data\dataset_division.txt'  # Path to the dataset division text file
    dataset_division_from_txt(txt_path)  # Call the function to process the dataset
