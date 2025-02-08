import os
import torch
import torch.utils
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy

# Define the InpaintingDataset class within the module
class InpaintingDataset(torch.utils.data.Dataset):

    def __init__(self, inputs_dir, masks_dir, image_transforms, masks_transforms):
        self.inputs_dir = inputs_dir
        self.masks_dir = masks_dir
        self.input_files = sorted(os.listdir(inputs_dir))
        print("Number of input images:", len(self.input_files))
        self.image_transforms = image_transforms
        self.masks_transforms = masks_transforms
        self.mask_files = sorted(os.listdir(self.masks_dir))
        self.length_masks = len(self.mask_files)
        print("Number of masks:", self.length_masks)

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

        return masked_image, mask, input_image 

# Load Dataset: prepara DataLoader para el batch training. INPUT are resized to (3, img_size, img_size)
def load_dataset(inputs_dir, masks_dir, batch_size=4, img_size=512, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):

    assert abs((train_ratio + val_ratio + test_ratio) - 1) < 1e-5, "The sum of train_ratio, val_ratio, and test_ratio must be equal to 1."

    images_transforms = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    masks_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size),
        transforms.ToTensor()
    ])

    full_dataset = InpaintingDataset(
        inputs_dir=inputs_dir,
        masks_dir=masks_dir,
        image_transforms=images_transforms,
        masks_transforms=masks_transforms
    )
    
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Merge Inputs and Masks: combina img de habitación vacia con mascara de habitación vacia
def merge_image_with_mask(input_image, mask_image):
    # Convert tensors to numpy arrays for manipulation
    input_image = input_image.cpu().detach().numpy().transpose(1, 2, 0)
    mask_image = mask_image.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Convert to [0, 255] range
    output_image = ((input_image + 1) * 127.5).astype(numpy.uint8)
    
    # Apply mask
    output_image[mask_image.squeeze() > 0] = 0
    
    # Convert back to tensor in range [-1, 1]
    output_tensor = torch.from_numpy(output_image).permute(2, 0, 1).float() / 127.5 - 1

    from image_service import save_image
    save_image(output_tensor, "C:/temp/masked-image.png")
    
    return output_tensor
