import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import os

class VolumeViewer:
    def __init__(self, volume, click_radius=5):
        """
        Classe per visualizzare un volume 3D e selezionare/rimuovere punti in diverse sezioni.

        :param volume: array 3D (H, W, D). Se il volume è un array NumPy, viene convertito in un array CuPy.
        :param click_radius: Distanza massima in pixel per rilevare un click vicino a un punto (default: 5)
        """
        # Se il volume non è già un array CuPy, lo converto
        if not isinstance(volume, cp.ndarray):
            self.volume = cp.asarray(volume)
        else:
            self.volume = volume

        self.slice_idx = 0  # Indice iniziale della slice
        self.clicked_points = []  # Lista per memorizzare i punti selezionati
        self.click_radius = click_radius  # Raggio di click per cancellare i punti
        self.view_mode = 'XY'  # Modalità di visualizzazione corrente: 'XY', 'ZY', 'XZ'
        self.unit_vectors = []
        self.img = None

        # Crea la figura
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)

        # Visualizza la slice iniziale
        self.update_image()

        # Aggiunge una barra dei colori
        self.cbar = self.fig.colorbar(self.img, ax=self.ax)
        self.cbar.set_label("Intensity")

        # Aggiunge uno slider per navigare tra le slice
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Slice', 0, self.get_max_slices() - 1, valinit=self.slice_idx, valstep=1)
        self.slider.on_changed(self.update_slice)

        # Aggiunge bottoni per cambiare il piano di visualizzazione
        ax_button_xy = plt.axes([0.1, 0.2, 0.15, 0.05])
        ax_button_zy = plt.axes([0.4, 0.2, 0.15, 0.05])
        ax_button_xz = plt.axes([0.7, 0.2, 0.15, 0.05])

        self.button_xy = Button(ax_button_xy, 'XY Plane')
        self.button_zy = Button(ax_button_zy, 'ZY Plane')
        self.button_xz = Button(ax_button_xz, 'XZ Plane')

        self.button_xy.on_clicked(lambda event: self.change_view('XY'))
        self.button_zy.on_clicked(lambda event: self.change_view('ZY'))
        self.button_xz.on_clicked(lambda event: self.change_view('XZ'))

        # Connette l'evento di click del mouse
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def get_max_slices(self):
        """Restituisce il numero massimo di slice per la modalità di visualizzazione corrente."""
        if self.view_mode == 'XY':
            return self.volume.shape[2]
        elif self.view_mode == 'ZY':
            return self.volume.shape[0]
        elif self.view_mode == 'XZ':
            return self.volume.shape[1]

    def update_image(self):
        """Aggiorna l'immagine visualizzata in base al piano corrente."""
        self.ax.clear()

        if self.view_mode == 'XY':
            # Converte la slice da CuPy a NumPy per Matplotlib
            img_data = cp.asnumpy(self.volume[:, :, self.slice_idx])
        elif self.view_mode == 'ZY':
            img_data = cp.asnumpy(self.volume[self.slice_idx, :, :].T)  # Trasponi per l'allineamento
        elif self.view_mode == 'XZ':
            img_data = cp.asnumpy(self.volume[:, self.slice_idx, :])

        self.img = self.ax.imshow(img_data, cmap='gray', vmin=0, vmax=255)
        self.ax.set_title(f"Slice: {self.slice_idx} ({self.view_mode} plane)")

        self.redraw_points()
        self.fig.canvas.draw_idle()

    def update_slice(self, val):
        """Aggiorna la slice visualizzata quando il valore dello slider cambia."""
        self.slice_idx = int(self.slider.val)
        self.update_image()

    def onclick(self, event):
        """Gestisce il click del mouse per aggiungere/rimuovere punti."""
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            print("Click:", x, y)

            # Converte (x, y) in coordinate (x, y, z) a seconda del piano
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

            # Verifica se il click è vicino ad un punto esistente
            for i, (px, py, pz) in enumerate(self.clicked_points):
                if (abs(pz - z) <= self.click_radius and
                    abs(px - x) <= self.click_radius and
                    abs(py - y) <= self.click_radius):
                    print(f"Punto rimosso: (x={px}, y={py}, z={pz})")
                    del self.clicked_points[i]
                    self.redraw_points()
                    return

            # Altrimenti, aggiunge il nuovo punto
            self.clicked_points.append((x, y, z))
            print(f"Punto aggiunto: (x={x}, y={y}, z={z})")
            self.redraw_points()
            if len(self.clicked_points) == 2:
                self.calculate_unit_vectors()

    def redraw_points(self):
        """Ridisegna i punti selezionati nella slice corrente."""
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
        """Cambia il piano di visualizzazione tra XY, ZY ed XZ."""
        self.view_mode = mode
        self.slice_idx = 0  # Resetta l'indice della slice
        self.slider.valmax = self.get_max_slices() - 1
        self.slider.set_val(0)
        self.update_image()

    def get_selected_points(self):
        """Restituisce tutti i punti selezionati."""
        return self.clicked_points

    def reset_points(self):
        """Resetta la lista dei punti selezionati."""
        self.clicked_points = []
        self.redraw_points()

    def show(self):
        """Visualizza la figura."""
        plt.show()

    def calculate_unit_vectors(self):
        """
        Per ogni coppia di punti selezionati,
        calcola il vettore unitario che li unisce.
        """
        self.unit_vectors = []
        if len(self.clicked_points) < 2:
            print("Sono necessari almeno due punti per calcolare il vettore.")
            return

        for i in range(0, len(self.clicked_points) - 1, 2):
            # Converte le tuple in array CuPy
            p1 = cp.array(self.clicked_points[i])
            p2 = cp.array(self.clicked_points[i + 1])
            vector = p2 - p1
            norm = cp.linalg.norm(vector)
            if norm != 0:
                vector = vector / norm
            else:
                print("I punti sono identici; impossibile calcolare il vettore unitario.")
                continue
            self.unit_vectors.append(vector)

def visualize_image(image, points=None):
    """
    Displays a 2D image with optional points highlighted.
    If the image is a CuPy array, it is converted to a NumPy array.
    
    :param image: 2D array representing the image
    :param points: list of tuples (x, y) for the points to highlight in red
    """
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)
    
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.title("2D Image Visualization")
    plt.axis('off')
    
    # If points are provided, plot them in red
    if points:
        points = np.array(points)
        plt.scatter(points[:, 0], points[:, 1], c='red', marker='x')
    
    plt.show()

def save_image(image, points=None, save_folder="visualizations"):
    """
    Saves a 2D image with optional points highlighted to a specified folder.
    If the image is a CuPy array, it is converted to a NumPy array.
    
    :param image: 2D array representing the image
    :param points: list of tuples (x, y) for the points to highlight in red
    :param save_folder: folder where the image will be saved, default is 'visualizations'
    """
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)
    
    # Make sure the folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Generate an incremental file name
    existing_files = os.listdir(save_folder)
    num_images = len([f for f in existing_files if f.endswith('.png')])  # Count current PNG files
    save_path = os.path.join(save_folder, f"image_{num_images + 1}.png")
    
    # Create the plot
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.title("2D Image Visualization")
    plt.axis('off')
    
    # If points are provided, plot them in red
    if points:
        points = np.array(points)
        plt.scatter(points[:, 0], points[:, 1], c='red', marker='x')
    
    # Save the image
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to avoid displaying it in an interactive session
    
    print(f"Image saved to {save_path}")
