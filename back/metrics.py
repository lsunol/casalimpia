import torch
import torchmetrics
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import lpips

# Crear el objeto PSNR
psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().cpu()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Convierte a rango [-1, 1]
])

def calculate_psnr(generated_image, target_image):
    """
    Calcula el PSNR entre una imagen generada y su correspondiente imagen original.

    Args:
        generated_image (Union[PIL.Image, torch.Tensor, np.ndarray]): Imagen generada
        target_image (Union[PIL.Image, torch.Tensor]): Imagen original

    Returns:
        float: Valor de PSNR entre las dos imágenes
    """
    # Convert numpy array to tensor if needed
    if isinstance(generated_image, np.ndarray):
        generated_image = torch.from_numpy(generated_image).permute(2, 0, 1) / 255.0

    # Convert PIL images to tensors if needed
    if isinstance(generated_image, Image.Image):
        generated_image = transform(generated_image)
    if isinstance(target_image, Image.Image):
        target_image = transform(target_image)

    # Move to CPU and add batch dimension
    if torch.is_tensor(generated_image):
        generated_image = generated_image.detach().cpu().unsqueeze(0)
    if torch.is_tensor(target_image):
        target_image = target_image.detach().cpu().unsqueeze(0)
    
    # Ensure images are in range [0, 1]
    if (generated_image.min() < 0) or (generated_image.max() > 1):
        generated_image = (generated_image + 1) / 2
    if (target_image.min() < 0) or (target_image.max() > 1):
        target_image = (target_image + 1) / 2

    # Calculate PSNR
    psnr_value = psnr_metric(generated_image, target_image)
    return psnr_value.item()

def calculate_ssim(pred_image, target_image):
    """Calculate SSIM between predicted and target images
    
    Args:
        pred_image: Single PIL Image or torch tensor
        target_image: Single PIL Image or torch tensor
        
    Returns:
        float: SSIM value
    """
    # Convert PIL images to numpy arrays
    if isinstance(pred_image, Image.Image):
        pred_image = np.array(pred_image) / 255.0  # Normalize to [0,1]
    if isinstance(target_image, Image.Image):
        target_image = np.array(target_image) / 255.0  # Normalize to [0,1]
        
    # Handle torch tensors
    if torch.is_tensor(pred_image):
        pred_image = pred_image.cpu().numpy()
    if torch.is_tensor(target_image):
        target_image = target_image.cpu().numpy()
        
    # Handle channel first format
    if pred_image.shape[0] == 3:  # If channels first, move to last dimension
        pred_image = pred_image.transpose(1, 2, 0)
    if target_image.shape[0] == 3:
        target_image = target_image.transpose(1, 2, 0)
    
    # Ensure values are in range [0,1]
    if pred_image.max() > 1.0:
        pred_image = pred_image / 255.0
    if target_image.max() > 1.0:
        target_image = target_image / 255.0
        
    return ssim(pred_image, target_image, channel_axis=2, data_range=1.0)

def calculate_lpips(pred_image, target_image, device='cuda'):
    """Calculate LPIPS perceptual distance between predicted and target images
    
    Args:
        pred_image: Single PIL Image, numpy array or torch tensor 
        target_image: Single PIL Image, numpy array or torch tensor
        device: Device to run LPIPS model on
        
    Returns:
        float: LPIPS distance value (lower is better)
    """
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Convert PIL images to tensors
    if isinstance(pred_image, Image.Image):
        pred_image = transforms.ToTensor()(pred_image).unsqueeze(0)
    if isinstance(target_image, Image.Image):
        target_image = transforms.ToTensor()(target_image).unsqueeze(0)
        
    # Convert numpy arrays to tensors
    if isinstance(pred_image, np.ndarray):
        pred_image = torch.from_numpy(pred_image).permute(2, 0, 1).unsqueeze(0) / 255.0
    if isinstance(target_image, np.ndarray):
        target_image = torch.from_numpy(target_image).permute(2, 0, 1).unsqueeze(0) / 255.0
        
    # Ensure tensors are in right format
    if torch.is_tensor(pred_image):
        if pred_image.dim() == 3:
            pred_image = pred_image.unsqueeze(0)
        pred_image = pred_image.to(device)
    if torch.is_tensor(target_image):
        if target_image.dim() == 3:
            target_image = target_image.unsqueeze(0) 
        target_image = target_image.to(device)
    
    # Normalize to [-1,1] if in [0,1]
    if pred_image.min() >= 0 and pred_image.max() <= 1:
        pred_image = pred_image * 2 - 1
    if target_image.min() >= 0 and target_image.max() <= 1:
        target_image = target_image * 2 - 1
        
    with torch.no_grad():
        lpips_value = loss_fn(pred_image, target_image)
        
    return lpips_value.item()
