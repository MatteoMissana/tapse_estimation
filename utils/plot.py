import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
class VolumeViewer:
    def __init__(self, volume):
        """
        Class to visualize a 3D volume and select points with mouse clicks.

        :param volume: 3D NumPy array (H, W, D)
        """
        self.volume = volume
        self.slice_idx = 0  # Initial slice index
        self.clicked_points = []  # List to store selected points

        # Create the figure
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        # Display the initial slice
        self.img = self.ax.imshow(self.volume[:, :, self.slice_idx], cmap='gray', vmin=0, vmax=255)
        self.ax.set_title(f"Slice: {self.slice_idx}")

        # Add a colorbar
        self.cbar = self.fig.colorbar(self.img, ax=self.ax)
        self.cbar.set_label("Intensity")

        # Add a slider to navigate through slices
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Slice', 0, self.volume.shape[2] - 1, valinit=self.slice_idx, valstep=1)
        self.slider.on_changed(self.update_slice)

        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def update_slice(self, val):
        """Update the displayed slice when the slider value changes."""
        self.slice_idx = int(self.slider.val)
        self.img.set_data(self.volume[:, :, self.slice_idx])
        self.ax.set_title(f"Slice: {self.slice_idx}")
        self.redraw_points()
        self.fig.canvas.draw_idle()

    def onclick(self, event):
        """Handle mouse click to save (x, y, z) coordinates."""
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            z = self.slice_idx
            self.clicked_points.append((x, y, z))
            print(f"Point selected: (x={x}, y={y}, z={z})")
            self.redraw_points()

    def redraw_points(self):
        """Redraw the selected points on the current slice."""
        self.ax.clear()
        self.img = self.ax.imshow(self.volume[:, :, self.slice_idx], cmap='gray', vmin=0, vmax=255)
        self.ax.set_title(f"Slice: {self.slice_idx}")

        # Draw all selected points in the current slice
        points_in_slice = [(x, y) for x, y, z in self.clicked_points if z == self.slice_idx]
        if points_in_slice:
            x_vals, y_vals = zip(*points_in_slice)
            self.ax.scatter(x_vals, y_vals, c='red', marker='x', s=50)

        self.fig.canvas.draw_idle()

    def get_selected_points(self):
        """Return all selected points."""
        return self.clicked_points

    def reset_points(self):
        """Reset the list of selected points."""
        self.clicked_points = []
        self.redraw_points()

    def show(self):
        """Display the figure."""
        plt.show()


def visualize_image(image):
    # Visualize the image using imshow
    plt.imshow(image, cmap='gray')  # 'gray' cmap is for grayscale images
    plt.colorbar()  # Optional: to show a color scale bar
    plt.title("2D Image Visualization")  # Optional: Title for the image
    plt.axis('off')  # Optional: Turn off the axis labels
    plt.show()  # Show the image