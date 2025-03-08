import os
import torch
import torch.utils
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy 
from scipy.ndimage import binary_dilation


# Define the InpaintingDataset class within the module
class InpaintingDataset(torch.utils.data.Dataset):

    def __init__(self, inputs_dir, masks_dir, image_transforms, masks_transforms, mask_padding, logger):
        
        self.logger = logger
        self.inputs_dir = inputs_dir
        self.masks_dir = masks_dir
        self.input_files = sorted(os.listdir(inputs_dir))
        logger.info(f"Number of input images: {len(self.input_files)}")
        self.image_transforms = image_transforms
        self.masks_transforms = masks_transforms
        self.mask_files = sorted(os.listdir(self.masks_dir))
        self.length_masks = len(self.mask_files)
        logger.info(f"Number of masks: {self.length_masks}")
        self.mask_padding = mask_padding

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.
        This method retrieves an image pair and a random mask from the directory. 
        It loads the input image, applies the random mask, and returns
        the masked image along with the original mask and input image.
        Args:
            idx (int): Index to retrieve the item from dataset.
        Returns:
            tuple: A tuple containing:
                - masked_image (torch.Tensor): The input image merged with the mask
                - mask (torch.Tensor): The binary mask applied to the image
                - target_image (torch.Tensor): The original unmasked input image 
        Notes:
            - Input images are loaded as RGB
            - Masks are loaded as grayscale (L mode)
            - Both images and masks are transformed according to the dataset's transform parameters
            - Masks are cycled through using modulo operation if there are fewer masks than images
        """
        # Load input image
        input_path = os.path.join(self.inputs_dir, self.input_files[idx])
        input_image = Image.open(input_path).convert("RGB")
        input_image = self.image_transforms(input_image)

        # Pick one random mask file from directory
        random_mask_idx = torch.randint(0, self.length_masks, (1,)).item()
        mask_path = os.path.join(self.masks_dir, self.mask_files[random_mask_idx])
        mask = Image.open(mask_path).convert("L")
        mask = self.masks_transforms(mask)

        masked_image = merge_image_with_mask(input_image, mask)

        return masked_image, add_padding_to_mask(mask, self.mask_padding), input_image, mask 

# Load Dataset: prepara DataLoader para el batch training. INPUT are resized to (3, img_size, img_size)
def load_dataset(inputs_dir, masks_dir, logger, batch_size=4, img_size=512, mask_padding=0, train_ratio=1, val_ratio=0, test_ratio=0, seed=42):

    assert abs((train_ratio + val_ratio + test_ratio) - 1) < 1e-5, "The sum of train_ratio, val_ratio, and test_ratio must be equal to 1."

    images_transforms_random_crop = transforms.Compose([                                                    
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),        # Redimensiona a (512, 512)
        transforms.RandomCrop(img_size),                                                        # Corta una región de 512x512
        transforms.ToTensor(),                                                                  # Convierte a tensor: [3, 512, 512] y escala los píxeles a [0,1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])                         # Normaliza a [-1,1]
    ])                                                                                          # Resultado: Tensor de forma [3, 512, 512], tipo torch.float32, con valores en [-1, 1].

    masks_transforms_random_crop = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size),
        transforms.ToTensor()
    ])

    # Check if inputs_dir contains required folders
    required_folders = ['train', 'validation', 'test']
    existing_folders = os.listdir(inputs_dir)
    missing_folders = [folder for folder in required_folders if folder not in existing_folders]
    if missing_folders:
        raise ValueError(f"Missing required folders in {inputs_dir}: {', '.join(missing_folders)}")

    # Create the full dataset with the new transforms
    train_dataset = InpaintingDataset(
        inputs_dir=inputs_dir + '/train',
        masks_dir=masks_dir,
        mask_padding=mask_padding,
        image_transforms=images_transforms_random_crop,
        masks_transforms=masks_transforms_random_crop,
        logger=logger
    )
    
    val_dataset = InpaintingDataset(
        inputs_dir=inputs_dir + '/validation',
        masks_dir=masks_dir,
        mask_padding=mask_padding,
        image_transforms=images_transforms_random_crop,
        masks_transforms=masks_transforms_random_crop,
        logger=logger
    )
    
    test_dataset = InpaintingDataset(
        inputs_dir=inputs_dir + '/test',
        masks_dir=masks_dir,
        mask_padding=mask_padding,
        image_transforms=images_transforms_random_crop,
        masks_transforms=masks_transforms_random_crop,
        logger=logger
    )

    logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # Create a small dataset with just the first image for sampling/testing
    images_transforms_center_crop = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    masks_transforms_center_crop = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    overfitting_dataset = InpaintingDataset(
        inputs_dir="./back/data/singleImageDataset/emptyRoom",
        masks_dir="./back/data/singleImageDataset/emptyMask",
        mask_padding=mask_padding,
        image_transforms=images_transforms_center_crop,
        masks_transforms=masks_transforms_center_crop,
        logger=logger
    )

    sampling_dataset = InpaintingDataset(
        inputs_dir="./back/data/samplingDataset/",
        masks_dir="./back/data/singleImageDataset/emptyMask",
        mask_padding=mask_padding,
        image_transforms=images_transforms_center_crop,
        masks_transforms=masks_transforms_center_crop,
        logger=logger
    )

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    overfitting_loader = torch.utils.data.DataLoader(overfitting_dataset, batch_size=max(batch_size, len(overfitting_dataset)), shuffle=False)
    sampling_loader = torch.utils.data.DataLoader(sampling_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader, overfitting_loader, sampling_loader

# Merge Inputs and Masks: combina img de habitación vacia con mascara de habitación vacia
def merge_image_with_mask(input_image, mask_image):
    # Convert tensors to numpy arrays for manipulation
    input_image = input_image.cpu().detach().numpy().transpose(1, 2, 0)
    mask_image = mask_image.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Convert to [0, 255] range
    output_image = ((input_image + 1) * 127.5).astype(numpy.uint8)
    
    # Apply black mask (R=0, G=0, B=0)
    mask = mask_image.squeeze() > 0.5
    output_image[mask, 0] = 0    # Red channel
    output_image[mask, 1] = 0    # Green channel
    output_image[mask, 2] = 0    # Blue channel
    
    # Convert back to tensor in range [-1, 1]
    output_tensor = torch.from_numpy(output_image).permute(2, 0, 1).float() / 127.5 - 1

    return output_tensor

def add_padding_to_mask(mask, num_pixels=5):
    """
    Expande la región enmascarada en una máscara binaria.

    Args:
        mask (torch.Tensor): Máscara en formato tensor, valores en [0,1], tamaño (1, H, W).
        num_pixels (int): Número de píxeles a expandir la máscara.

    Returns:
        torch.Tensor: Máscara con padding aplicado.
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError("La máscara debe ser un tensor de PyTorch.")
    
    # Convertir el tensor de PyTorch a numpy
    mask_np = mask.squeeze(0).cpu().numpy()  # (H, W)

    # Asegurar que es una máscara binaria
    mask_np = (mask_np > 0.5).astype(numpy.uint8)

    # Aplicar dilatación morfológica para expandir la máscara
    structuring_element = numpy.ones((num_pixels * 2 + 1, num_pixels * 2 + 1), dtype=numpy.uint8)
    expanded_mask_np = binary_dilation(mask_np, structure=structuring_element).astype(numpy.float32)

    # Convertir de nuevo a tensor y mantener el formato (1, H, W)
    expanded_mask = torch.from_numpy(expanded_mask_np).unsqueeze(0)

    return expanded_mask

def save_tensor_as_grayscale_png(tensor, output_path):
    """
    Guarda un tensor de tamaño 1x512x512 (o cualquier tensor 2D/3D con un solo canal)
    como una imagen PNG en escala de grises. Util para guardar máscaras.
    
    Args:
        tensor (torch.Tensor): Tensor de entrada, debe ser de forma [1, H, W] o [H, W]
        output_path (str): Ruta donde se guardará la imagen PNG
    
    Returns:
        None
    """
    # Asegurarse de que el tensor esté en CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Manejar diferentes formas de tensor
    if tensor.dim() == 3 and tensor.size(0) == 1:
        # Si es [1, H, W], quitar la primera dimensión
        tensor = tensor.squeeze(0)
    elif tensor.dim() != 2:
        raise ValueError(f"El tensor debe ser de forma [1, H, W] o [H, W], pero tiene forma {tensor.shape}")
    
    # Convertir a NumPy
    numpy_array = tensor.detach().numpy()
    
    # Normalizar a rango [0, 255]
    min_val = numpy_array.min()
    max_val = numpy_array.max()
    if min_val != max_val:  # Evitar división por cero
        numpy_array = ((numpy_array - min_val) / (max_val - min_val) * 255).astype(numpy.uint8)
    else:
        numpy_array = numpy.zeros_like(numpy_array, dtype=numpy.uint8)
    
    # Crear una imagen PIL
    img = Image.fromarray(numpy_array, mode='L')  # 'L' indica modo escala de grises
    
    # Asegurarse de que el directorio existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar la imagen
    img.save(output_path)
