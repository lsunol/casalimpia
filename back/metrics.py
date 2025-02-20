import torch
import torchmetrics
import torchvision.transforms as transforms
from PIL import Image

# Crear el objeto PSNR
psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().cpu()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Convierte a rango [-1, 1]
])

def calculate_psnr(generated_images, target_image):
    """
    Calcula el PSNR entre un array de imágenes generadas y su correspondiente imagen original.

    Args:
        generated_images (List[PIL.Image]): Array de imágenes generadas
        target_image (Union[torch.Tensor, PIL.Image]): Imagen original. Si es tensor (3, H, W) en [-1, 1].

    Returns:
        float: Valor de PSNR entre las dos imágenes.
    """
    # Convertir array de PIL images a un único tensor
    if isinstance(generated_images[0], Image.Image):
        generated_tensors = []
        for img in generated_images:
            generated_tensors.append(transform(img))
        generated_image = torch.stack(generated_tensors)
    else:
        generated_image = generated_images

    # Convertir target a tensor si es necesario
    if isinstance(target_image, Image.Image):
        target_image = transform(target_image)

    # Mover todo a CPU
    generated_image = generated_image.detach().cpu()
    target_image = target_image.detach().cpu()
    
    # Asegurar que las imágenes están en el rango [0, 1]
    if (generated_image.min() < 0) or (generated_image.max() > 1):
        generated_image = (generated_image + 1) / 2

    if (target_image.min() < 0) or (target_image.max() > 1):
        target_image = (target_image + 1) / 2

    # Añadir dimensión de batch al target si no está presente
    if len(target_image.shape) == 3:
        target_image = target_image.unsqueeze(0)
        # Repetir target para match con el número de imágenes generadas
        target_image = target_image.repeat(len(generated_image), 1, 1, 1)

    # Calcular PSNR
    psnr_value = psnr_metric(generated_image, target_image)
    return psnr_value.item()  # Convertir a valor escalar