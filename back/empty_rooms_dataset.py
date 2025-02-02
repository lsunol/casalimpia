import os
import torch
import torchvision.transforms as transforms
from PIL import Image

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

        input_path = os.path.join(self.inputs_dir, self.input_files[idx])
        input_image = Image.open(input_path).convert("RGB")
        input_image = self.image_transforms(input_image)

        # Pick one random mask file from directory
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx % self.length_masks])
        mask_image = Image.open(mask_path).convert("L")
        mask_image = self.masks_transforms(mask_image)

        # target_image is still the same as input_image
        return input_image, mask_image, input_image 

# Load Dataset: prepara DataLoader para el batch training. INPUT are resized to (3, img_size, img_size)
def load_dataset(inputs_dir, masks_dir, batch_size=4, img_size=512):

    images_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    masks_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    dataset = InpaintingDataset(
        inputs_dir=inputs_dir,
        masks_dir=masks_dir,
        image_transforms=images_transforms,
        masks_transforms=masks_transforms
    )
    
    dataset = InpaintingDataset(inputs_dir, masks_dir, images_transforms, masks_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# Merge Inputs and Masks: combina img de habitación vacia con mascara y salva el conjunto en "output_dir"
# def merge_inputs_and_masks(empty_rooms_dir, masks_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     empty_room_files = sorted(os.listdir(empty_rooms_dir))
#     mask_files = sorted(os.listdir(masks_dir))

#     for room_file, mask_file in zip(empty_room_files, mask_files):
#         room_path = os.path.join(empty_rooms_dir, room_file)
#         mask_path = os.path.join(masks_dir, mask_file)

#         empty_room = Image.open(room_path).convert("RGB")
#         mask_data = numpy.load(mask_path)
#         mask = numpy.any(mask_data["masks"], axis=0) if len(mask_data["masks"].shape) == 3 else mask_data["masks"]      # Combine masks
#         mask_image = Image.fromarray((mask * 255).astype(numpy.uint8)).convert("L")

#         # Resize mask to match the room size
#         mask_image = mask_image.resize(empty_room.size, Image.Resampling.NEAREST)

#         # Combine room and mask
#         mask_overlay = Image.new("RGB", empty_room.size)                                        # Create an overlay
#         mask_overlay.paste(empty_room, (0, 0))                                                  # Paste the empty room image
#         red_layer = Image.new("RGB", empty_room.size, (255, 0, 0))                              # Red overlay for the mask
#         black_layer = Image.new("RGB", empty_room.size, (0, 0, 0))                              # Red overlay for the mask
#         mask_overlay.paste(black_layer, (0, 0), mask_image)                                     # Add the mask as a red overlay
#         mask_overlay = Image.blend(empty_room, mask_overlay, alpha=1.0)                         # Blend the overlay with the room image

#         # Save merged input
#         output_path = os.path.join(output_dir, room_file)
#         mask_overlay.save(output_path)
#         print(f"Saved merged input: {output_path}")

