import torch
import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def save_epoch_sample(input_image, input_mask, inferred_image, target_image, epoch, sample_index, output_path):
    """
    Crea una imagen con un título y concatena las imágenes proporcionadas.

    Parámetros:
        input_image (Union[PIL.Image, np.ndarray]): Imagen de entrada
        input_mask (Union[PIL.Image, np.ndarray]): Máscara de entrada
        inferred_image (Union[PIL.Image, np.ndarray]): Imagen inferida
        target_image (Union[PIL.Image, np.ndarray]): Imagen objetivo
        epoch (int): Número de epoch actual
        output_path (str): Ruta donde se guardará la imagen resultante
    """
    # Convert numpy arrays to PIL Images if necessary
    if isinstance(inferred_image, np.ndarray):
        inferred_image = Image.fromarray(inferred_image.astype(np.uint8))
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image.astype(np.uint8))
    if isinstance(input_mask, np.ndarray):
        input_mask = Image.fromarray(input_mask.astype(np.uint8))
    if isinstance(target_image, np.ndarray):
        target_image = Image.fromarray(target_image.astype(np.uint8))

    # Tamaño de las imágenes (asumimos que todas tienen el mismo tamaño)
    img_width, img_height = input_image.size
    image_margin = 10  # Margen entre las imágenes

    # Crear una imagen en blanco para el título
    title_height = 50  # Altura del título
    title_image = Image.new("RGB", (img_width * 3 + image_margin * 4, title_height), color="white")

    # Dibujar el título en la imagen
    draw = ImageDraw.Draw(title_image)
    try:
        font = ImageFont.truetype("arial.ttf", 40)  # Usar la fuente Arial
    except:
        font = ImageFont.load_default()  # Usar la fuente predeterminada

    text = f"Epoch {epoch + 1}:"
    _, _, text_width, text_height = font.getbbox(text)
    draw.text((image_margin, (title_height - text_height) // 2), text, font=font, fill="black")

    # Crear una imagen compuesta con las 3 columnas
    composite_width = img_width * 3 + image_margin * 4
    composite_height = img_height + image_margin
    composite_image = Image.new("RGB", (composite_width, composite_height), color="white")

    # Pegar las imágenes en la imagen compuesta
    composite_image.paste(input_image.convert("RGB"), (image_margin, 0))  # Columna 1: Imagen original con máscara
    composite_image.paste(inferred_image.convert("RGB"), (img_width + image_margin * 2, 0))  # Columna 2: Imagen inferida
    composite_image.paste(target_image.convert("RGB"), (img_width * 2 + image_margin * 3, 0))  # Columna 3: Imagen target

    # Crear la imagen final con el título y la imagen compuesta
    final_image = Image.new("RGB", (composite_width, title_height + composite_height))
    final_image.paste(title_image, (0, 0))
    final_image.paste(composite_image, (0, title_height))

    image_path = output_path + "sample_" + str(sample_index) + ".png"

    # Check if image already exists
    if os.path.exists(image_path):
        # Open existing image
        existing_image = Image.open(image_path)
        # Create new image with combined height
        combined_image = Image.new('RGB', (composite_width, existing_image.height + final_image.height))
        # Paste existing image at top
        combined_image.paste(existing_image, (0, 0))
        # Paste new image at bottom
        combined_image.paste(final_image, (0, existing_image.height))
        # Save combined image
        combined_image.save(image_path)
    else:
        # Save new image if no existing file
        final_image.save(image_path)

def save_image(image, path):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy().transpose(1, 2, 0)
    elif isinstance(image, PIL.Image.Image):
        image = np.array(image)
    image = ((image + 1) * 127.5).astype(np.uint8)
    Image.fromarray(image).save(path)

if __name__ == "__main__":

    # Tamaño de las imágenes
    img_size = (512, 512)

    # Crear una imagen sólida de color azul (input_image)
    input_image = Image.new("RGB", img_size, color="blue")

    # Crear una máscara con un círculo y ruido
    mask = Image.new("L", img_size, color=0)  # Máscara en escala de grises (inicialmente negra)
    draw = ImageDraw.Draw(mask)

    # Dibujar un círculo en el centro de la máscara
    circle_radius = 100
    circle_center = (img_size[0] // 2, img_size[1] // 2)
    draw.ellipse(
        [
            circle_center[0] - circle_radius,
            circle_center[1] - circle_radius,
            circle_center[0] + circle_radius,
            circle_center[1] + circle_radius,
        ],
        fill=255,  # Blanco (área de la máscara)
    )

    # Añadir ruido a la máscara
    mask_array = np.array(mask)
    noise = np.random.randint(0, 50, img_size, dtype=np.uint8)  # Ruido aleatorio
    mask_array = np.clip(mask_array + noise, 0, 255)  # Aplicar ruido y asegurar que los valores estén entre 0 y 255
    input_mask = Image.fromarray(mask_array, mode="L")

    # Crear una imagen sólida de color rojo (inferred_image)
    inferred_image = Image.new("RGB", img_size, color="red")

    # Crear una imagen sólida de color amarillo (target_image)
    target_image = Image.new("RGB", img_size, color="yellow")

    # Ruta de salida para la imagen
    output_path = "back/data/lora_trains/image_sandbox/"

    # Llamar al método para crear y guardar la imagen
    save_epoch_sample(
        input_image=input_image,
        input_mask=input_mask,
        inferred_image=inferred_image,
        target_image=target_image,
        epoch=1,
        output_path=output_path
    )

    # Llamar al método para crear y guardar la imagen
    save_epoch_sample(
        input_image=input_image,
        input_mask=input_mask,
        inferred_image=inferred_image,
        target_image=target_image,
        epoch=2,
        output_path=output_path
    )
