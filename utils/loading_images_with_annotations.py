import numpy as np
import matplotlib.pyplot as plt
import os

'''image visualization with annotations superimposed'''

def load_and_plot_annotations(file_path, annotation_path):
    """
    Load and display an image with annotations.
    
    Parameters:
    -----------
    file_path : str
        Path to the `.npz` file containing the images.
    annotation_path : str
        Path to the `.npz` file containing the annotations.
    """
    # Load images
    data = np.load(file_path)
    frames = data['video']  # Assume images are stored under the key 'video'
    
    # Load annotations
    annotations = np.load(annotation_path)['annotations']
    
    num_frames = frames.shape[2]
    
    plt.ion()  # Interactive mode
    
    for idx in range(num_frames):
        plt.clf()
        plt.imshow(frames[:, :, idx], cmap='gray')  # Display the image
        
        # Plot annotations
        for j in range(2):
            plt.scatter(annotations[idx][j][0], annotations[idx][j][1], 
                        color='r', marker='*', s=100)  # Annotations
        
        plt.title(f"Frame {idx + 1}/{num_frames}")
        plt.pause(0.5)  # Pause to visualize the frame
    
    plt.ioff()
    plt.show()


def visualize_dataset(dataset_path):
    """
    Visualize images and keypoints from a dataset.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the `.npz` file containing the dataset.
    """
    # Load the dataset
    dataset = np.load(dataset_path)
    images = dataset['images']
    keypoints = dataset['keypoints']
    
    num_images = images.shape[0]
    print(images.shape)
    
    plt.ion()
    plt.ion()  # Interactive mode

    for idx in range(num_images):
        plt.clf()
        plt.imshow(images[idx, :, :], cmap='gray')  # Display the image
        
        # Plot annotations
        for j in range(2):
            plt.scatter(keypoints[idx][j][0], keypoints[idx][j][1], 
                        color='r', marker='*', s=100)  # Annotations
        
        plt.title(f"image {idx + 1}/{num_images}")
        plt.pause(0.5)  # Pause to visualize the frame
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    dataset_path = 'data/dataset/test.npz'
    visualize_dataset(dataset_path)

