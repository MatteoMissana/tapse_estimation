import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def view_volume(volume):
    """
    View slices of a 3D volume (shape: H, W, D) using a Matplotlib slider.
    :param volume: 3D NumPy array (H, W, D)
    """
    # Set up the initial plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Initial slice index
    slice_idx = 0
    img = ax.imshow(volume[:, :, slice_idx], cmap='gray')
    ax.set_title(f"Slice: {slice_idx}")

    # Set up the slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # [x, y, width, height]
    slider = Slider(ax_slider, 'Slice', 0, volume.shape[2] - 1, valinit=slice_idx, valstep=1)

    # Update function for slider
    def update(val):
        idx = int(slider.val)
        img.set_data(volume[:, :, idx])
        ax.set_title(f"Slice: {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def visualize_image(image):
    # Visualize the image using imshow
    plt.imshow(image, cmap='gray')  # 'gray' cmap is for grayscale images
    plt.colorbar()  # Optional: to show a color scale bar
    plt.title("2D Image Visualization")  # Optional: Title for the image
    plt.axis('off')  # Optional: Turn off the axis labels
    plt.show()  # Show the image