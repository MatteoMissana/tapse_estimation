import numpy as np
import matplotlib.pyplot as plt
import os

def press(event):
    global user_input
    user_input = event.key

def annotate_2_points_2d_video(file_path):
    """
    Displays frames from a 2D ultrasound video stored in an `.npz` file and allows the user 
    to manually annotate two landmark points per frame. The annotated points are then saved 
    as a new `.npz` file.

    Parameters:
    -----------
    file_path : str
        Path to the `.npz` file containing the ultrasound frames.
        The `.npz` file should contain a numpy array with key 'arr_0' 
        representing a grayscale video of shape (height, width, num_frames).

    Functionality:
    --------------
    - Iterates through the frames of the ultrasound video.
    - Allows the user to manually select **two landmark points** per frame using mouse clicks.
    - Provides keyboard controls for navigation:
        - `Enter` → Select and save 2 points for the current frame.
        - `Left/Right arrow` → Move to the previous/next frame.
        - `Up/Down arrow` → Jump 10 frames forward/backward.
        - `C` → Save and close the annotation session.
    - Saves the annotated points as a new `.npz` file with the same name, 
      appending `_annotations.npz`.

    Output:
    -------
    - A new `.npz` file is created with two arrays:
        - `'arr_0'`: Original video frames.
        - `'annotations'`: An array of shape (num_frames, 2, 2) storing 
          the (x, y) coordinates of the two selected landmarks for each frame.
    """
    # Load the numpy array from `.npz`
    data = np.load(file_path)
    frames = data['video']  # Shape: (dim1, dim2, frame_index)

    # Output directory to save annotations
    save_path = file_path.replace('.npz', '_annotations.npz')

    # Initialize landmark coordinates (2 landmarks per frame)
    num_frames = frames.shape[2]
    ref_coord = np.zeros((num_frames, 2, 2))  # Shape: (frames, landmarks, (x, y))

    plt.ion()  # Enable interactive mode

    idx = 0  # Start at first frame
    idx_max = num_frames - 1  # Last frame index

    while True:
        plt.clf()
        plt.imshow(frames[:, :, idx], cmap='gray')  # Display current frame
        fig = plt.gcf()
        fig.set_size_inches(10, 10)

        # Plot previous frame's landmarks in blue, current frame's in red
        prev_idx = idx_max if idx == 0 else idx - 1
        for j in range(2):
            plt.scatter(ref_coord[idx][j][0], ref_coord[idx][j][1], color='r', marker='*', s=100)  # Current frame
            plt.scatter(ref_coord[prev_idx][j][0], ref_coord[prev_idx][j][1], color='b', marker='*', s=100)  # Previous frame

        plt.gcf().canvas.mpl_connect('key_press_event', press)

        print(f"Frame {idx + 1}/{num_frames}")
        print("Enter = correct landmarks, c = close\n")

        while not plt.waitforbuttonpress(100000):
            pass

        if user_input == "enter":
            coordinates = plt.ginput(n=2, timeout=0, show_clicks=True)  # Only 2 landmarks
            ref_coord[idx] = np.array(coordinates)
        elif user_input == "c":
            print("Saving annotations and closing...")
            np.savez(save_path, frames=frames, annotations=ref_coord)
            break
        elif user_input == "left":
            idx = idx_max if idx == 0 else idx - 1
        elif user_input == "right":
            idx = 0 if idx == idx_max else idx + 1
        elif user_input == "down":
            idx = max(0, idx - 10)
        elif user_input == "up":
            idx = min(idx_max, idx + 10)

    plt.ioff()
    plt.close()


if __name__ == "__main__":
    folder_path = r"D:\mmissana\data\best_slices"
    for subfolder in os.listdir(folder_path):
        folder = os.path.join(folder_path, subfolder)
        npz_path = os.path.join(folder, 'video_best_slice.npz')
        new_npz_path = os.path.join(folder, 'video_best_slice_annotations.npz')

        # **Check if file exists before loading**
        if not os.path.exists(npz_path) or os.path.exists(new_npz_path):
            print(f"Skipping {subfolder}, video_best_slice.npz not found or annotation has already been done.")
            continue

        # **Load using NumPy**
        annotate_2_points_2d_video(npz_path)