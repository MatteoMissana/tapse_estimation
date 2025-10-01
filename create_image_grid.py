from PIL import Image, ImageDraw, ImageFont

def create_grid_with_labels(image_paths, output_path="grid.png"):
    if len(image_paths) != 9:
        raise ValueError("You must provide exactly 9 image paths.")

    # Dimensioni singola immagine
    w, h = 369, 369
    grid_w, grid_h = 3 * w, 3 * h

    # Canvas vuoto
    grid_img = Image.new("RGB", (grid_w, grid_h))

    # Font
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except OSError:
        font = ImageFont.load_default()

    labels = ["100", "111", "140", "149", "160", "170", "190", "198", "199"]

    for idx, path in enumerate(image_paths):
        img = Image.open(path).resize((w, h))

        # Disegna etichetta
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), labels[idx], font=font, fill="white")

        # Posizione nella griglia 3x3
        x = (idx % 3) * w
        y = (idx // 3) * h
        grid_img.paste(img, (x, y))

    grid_img.save(output_path)
    print(f"Grid saved to {output_path}")

# Example usage
images = [
    '2d/results/boxplots_2/100/P4297P80_interpolated/image_1.png',
    '2d/results/boxplots_2/111/P429AL06_interpolated/image_1.png',
    '2d/results/boxplots_2/140/P429BN8C_interpolated/image_1.png',
    '2d/results/boxplots_2/149/P429D0OG_interpolated/image_1.png',
    '2d/results/boxplots_2/160/P429DN8I_interpolated/image_1.png',
    '2d/results/boxplots_2/170/P429EEOK_interpolated/image_1.png',
    '2d/results/boxplots_2/190/P429G08O_interpolated/image_1.png',
    '2d/results/boxplots_2/198/P429G9OQ_interpolated/image_1.png',
    '2d/results/boxplots_2/199/P42A0G2A_interpolated/image_1.png'
]

create_grid_with_labels(images, r"D:\mmissana\tapse_estimation\2d\results\boxplots_2\composition.png")
