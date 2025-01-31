import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class VolumeViewer:
    def __init__(self, volume, click_radius=5):
        """
        Class to visualize a 3D volume and select/remove points in different planes.

        :param volume: 3D NumPy array (H, W, D)
        :param click_radius: Maximum pixel distance to detect a click near a point (default: 5)
        """
        self.volume = volume
        self.slice_idx = 0  # Initial slice index
        self.clicked_points = []  # List to store selected points
        self.click_radius = click_radius  # Click radius to delete points
        self.view_mode = 'XY'  # Current view mode: 'XY', 'ZY', 'XZ'
        self.unit_vectors = []
        self.img = None

        # Create the figure
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)

        # Display the initial slice
        self.update_image()

        # Add a colorbar
        self.cbar = self.fig.colorbar(self.img, ax=self.ax)
        self.cbar.set_label("Intensity")

        # Add a slider to navigate through slices
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Slice', 0, self.get_max_slices() - 1, valinit=self.slice_idx, valstep=1)
        self.slider.on_changed(self.update_slice)

        # Add buttons to switch view mode
        ax_button_xy = plt.axes([0.1, 0.2, 0.15, 0.05])
        ax_button_zy = plt.axes([0.4, 0.2, 0.15, 0.05])
        ax_button_xz = plt.axes([0.7, 0.2, 0.15, 0.05])

        self.button_xy = Button(ax_button_xy, 'XY Plane')
        self.button_zy = Button(ax_button_zy, 'ZY Plane')
        self.button_xz = Button(ax_button_xz, 'XZ Plane')

        self.button_xy.on_clicked(lambda event: self.change_view('XY'))
        self.button_zy.on_clicked(lambda event: self.change_view('ZY'))
        self.button_xz.on_clicked(lambda event: self.change_view('XZ'))

        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def get_max_slices(self):
        """Returns the max index for the current viewing mode."""
        if self.view_mode == 'XY':
            return self.volume.shape[2]
        elif self.view_mode == 'ZY':
            return self.volume.shape[0]
        elif self.view_mode == 'XZ':
            return self.volume.shape[1]

    def update_image(self):
        """Updates the displayed slice based on the selected view mode."""
        self.ax.clear()

        if self.view_mode == 'XY':
            img_data = self.volume[:, :, self.slice_idx]
        elif self.view_mode == 'ZY':
            img_data = self.volume[self.slice_idx, :, :].T  # Transpose to align correctly
        elif self.view_mode == 'XZ':
            img_data = self.volume[:, self.slice_idx, :]

        self.img = self.ax.imshow(img_data, cmap='gray', vmin=0, vmax=255)
        self.ax.set_title(f"Slice: {self.slice_idx} ({self.view_mode} plane)")

        self.redraw_points()
        self.fig.canvas.draw_idle()

    def update_slice(self, val):
        """Update the displayed slice when the slider value changes."""
        self.slice_idx = int(self.slider.val)
        self.update_image()

    def onclick(self, event):
        """Handle mouse click to add/remove points."""
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            print(x, y)

            # Convert (x, y) to (x, y, z) based on the view mode
            if self.view_mode == 'XY':
                z = self.slice_idx
            elif self.view_mode == 'ZY':
                z = y
                y = x
                x = self.slice_idx
            elif self.view_mode == 'XZ':
                z = x
                x = y
                y = self.slice_idx

            # Check if the click is near an existing point
            for i, (px, py, pz) in enumerate(self.clicked_points):
                if abs(pz - z) <= self.click_radius and abs(px - x) <= self.click_radius and abs(py - y) <= self.click_radius:
                    # Remove the point if it's within the click radius
                    print(f"Point removed: (x={px}, y={py}, z={pz})")
                    del self.clicked_points[i]
                    self.redraw_points()
                    return

            # Otherwise, add the new point
            self.clicked_points.append((x, y, z))
            print(f"Point added: (x={x}, y={y}, z={z})")
            self.redraw_points()
            if len(self.clicked_points) == 2:
                self.calculate_unit_vectors()
    def redraw_points(self):
        """Redraw the selected points on the current slice."""
        points_in_slice = []

        for x, y, z in self.clicked_points:
            if self.view_mode == 'XY' and z == self.slice_idx:
                points_in_slice.append((x, y))
            elif self.view_mode == 'ZY' and x == self.slice_idx:
                points_in_slice.append((y, z))
            elif self.view_mode == 'XZ' and y == self.slice_idx:
                points_in_slice.append((z, x))

        if points_in_slice:
            x_vals, y_vals = zip(*points_in_slice)
            self.ax.scatter(x_vals, y_vals, c='red', marker='x', s=50)

        self.fig.canvas.draw_idle()

    def change_view(self, mode):
        """Switch between XY, ZY, and XZ planes."""
        self.view_mode = mode
        self.slice_idx = 0  # Reset slice index
        self.slider.valmax = self.get_max_slices() - 1
        self.slider.set_val(0)
        self.update_image()

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

    def calculate_unit_vectors(self):
        '''
        for each couple of points selected,
        calculates the unit vector passing through them
        '''
        self.unit_vectors = []
        if len(self.clicked_points) < 2:
            print("At least two points are needed to compute vectors.")
            return

        for i in range(len(self.clicked_points)):
            if not i%2:
                p1 = np.array(self.clicked_points[i])
                p2 = np.array(self.clicked_points[i + 1])
                vector = p2 - p1
                vector = vector/np.linalg.norm(vector)
                self.unit_vectors.append(vector)


def visualize_image(image):
    # Visualize the image using imshow
    plt.imshow(image, cmap='gray')  # 'gray' cmap is for grayscale images
    plt.colorbar()  # Optional: to show a color scale bar
    plt.title("2D Image Visualization")  # Optional: Title for the image
    plt.axis('off')  # Optional: Turn off the axis labels
    plt.show()  # Show the image