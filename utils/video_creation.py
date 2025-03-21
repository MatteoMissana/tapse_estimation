import cv2
import os
import re

def create_video_from_images(folder_path, output_video, fps=30):
    """
    Creates a video from a folder of PNG images.
    
    :param folder_path: Path to the folder containing PNG images.
    :param output_video: Path to the output video file (e.g., 'output.mp4').
    :param fps: Frames per second for the output video.
    
    Image Naming Convention:
    - Images should be named in a way that ensures correct ordering when sorted alphabetically.
    - Example: 'frame_001.png', 'frame_002.png', ..., 'frame_100.png'
    """

    # Get a list of all PNG images in the folder
    images = [img for img in os.listdir(folder_path) if img.endswith(".png")]

    # Sort the images using a regex to extract the numeric part
    images.sort(key=lambda img: int(re.search(r'\d+', img).group()))
    
    # Check if there are any images in the folder
    if not images:
        print("No PNG images found in the folder.")
        return
    
    # Read the first image to determine the video resolution
    first_image = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, _ = first_image.shape  # Get image dimensions
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 output
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Loop through each image, read it, and write it to the video file
    for image in images:
        frame = cv2.imread(os.path.join(folder_path, image))
        video_writer.write(frame)  # Write the frame to the video
    
    # Release the video writer to finalize the file
    video_writer.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    folder= r'results/Unet_augm7_new_filter'
    output = r'D:\mmissana\best_predictions_video.mp4'
    create_video_from_images(folder, output, fps=3)