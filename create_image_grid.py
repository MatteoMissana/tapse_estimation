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
    r'C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\images_paper\images_3_2_pred_plus_annotation\100.png',
    r'C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\images_paper\images_3_2_pred_plus_annotation\111.png',
    r'C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\images_paper\images_3_2_pred_plus_annotation\140.png',
    r'C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\images_paper\images_3_2_pred_plus_annotation\149.png',
    r'C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\images_paper\images_3_2_pred_plus_annotation\160.png',
    r'C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\images_paper\images_3_2_pred_plus_annotation\170.png',
    r'C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\images_paper\images_3_2_pred_plus_annotation\190.png',
    r'C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\images_paper\images_3_2_pred_plus_annotation\198.png',
    r'C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\images_paper\images_3_2_pred_plus_annotation\199.png'
]

create_grid_with_labels(images, r"C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\mmissana\relevant_data\images_paper\images_3_2_pred_plus_annotation\grid.png")
