import numpy as np
import matplotlib.pyplot as plt
import os

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

if __name__ == "__main__":
    file_path = r"D:\mmissana\data\best_slices\811001\video_best_slice.npz"
    annotation_path = r"D:\mmissana\data\best_slices\811001\video_best_slice_annotations.npz"
    
    load_and_plot_annotations(file_path, annotation_path)
