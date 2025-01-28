import os, csv, torch, numpy, scipy.io, PIL.Image
import torchvision.transforms as transforms
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import cv2
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import torchvision

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Setup Inpainting Pipeline: CARGA DIFFUSSION MODEL y lo manda a GPU
def setup_inpainting_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
    ).to(device)
    return pipe


# Dataset Preparation: recibe directorio de imágenes, directorio de máscaras y transforms como entrada.
# Se transforma imagen a formato PyTorch (C, H, W) 
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

# Merge Inputs and Masks: combrina img de habitación vacia con mascara y salva el conjunto en "output_dir"
def merge_inputs_and_masks(empty_rooms_dir, masks_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)                              # Create the output directory if it doesn't exist

    empty_room_files = sorted(os.listdir(empty_rooms_dir))              # List of empty room images
    mask_files = sorted(os.listdir(masks_dir))                          # List of mask files

    for room_file, mask_file in zip(empty_room_files, mask_files):
        room_path = os.path.join(empty_rooms_dir, room_file)
        mask_path = os.path.join(masks_dir, mask_file)

        empty_room = Image.open(room_path).convert("RGB")                                                               # Load empty room image
        mask_data = numpy.load(mask_path)                                                                               # Load mask data
        mask = numpy.any(mask_data["masks"], axis=0) if len(mask_data["masks"].shape) == 3 else mask_data["masks"]      # Combine masks
        mask_image = Image.fromarray((mask * 255).astype(numpy.uint8)).convert("L")                                     # Convert mask to grayscale

        # Resize mask to match the room size
        mask_image = mask_image.resize(empty_room.size, Image.Resampling.NEAREST)

        # Combine room and mask
        mask_overlay = Image.new("RGB", empty_room.size)                                        # Create an overlay
        mask_overlay.paste(empty_room, (0, 0))                                                  # Paste the empty room image
        red_layer = Image.new("RGB", empty_room.size, (255, 0, 0))                              # Red overlay for the mask
        black_layer = Image.new("RGB", empty_room.size, (0, 0, 0))                              # Red overlay for the mask
        mask_overlay.paste(black_layer, (0, 0), mask_image)                                     # Add the mask as a red overlay
        mask_overlay = Image.blend(empty_room, mask_overlay, alpha=1.0)                         # Blend the overlay with the room image

        # Save merged input
        output_path = os.path.join(output_dir, room_file)
        mask_overlay.save(output_path)
        print(f"Saved merged input: {output_path}")


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

    dataset = InpaintingDataset(inputs_dir, masks_dir, images_transforms, masks_transforms)                             # Initialize dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)      # Create DataLoader
    return dataloader


# Setup LoRA Model
def setup_model_with_lora(model_id):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    # Inspect available modules in the UNet
    # print("Available modules in UNet:")                                                         
    supported_modules = []

    for name, module in pipe.unet.named_modules():
        # print(name)
        # Check if module is compatible with LoRA
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):                              
            supported_modules.append(name)

    # Ensure there are supported modules
    if not supported_modules:                                                                   
        raise ValueError("No compatible modules found for LoRA in the UNet.")

    # print(f"Using the following modules for LoRA: {supported_modules}")

    # Setup LoRA with compatible modules
    lora_config = LoraConfig(                                                                       
        r=16,
        lora_alpha=32,
        target_modules=supported_modules,
        lora_dropout=0.1
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    return pipe


# Training LoRA: concatenates images and masks into a single tensor for training (6 chanel input)
def train_lora(pipe, dataloader, num_epochs=5, lr=5e-5):
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)
    pipe.unet.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0

        for input_images, input_masks, targets in tqdm(dataloader):

            print("inputs shape:", input_images.shape)
            # Separate images and masks
            print("masks shape:", input_masks.shape)

            input_images = input_images.to(device).to(torch.float16)  # Image: 3 channels
            input_masks = input_masks.to(device).to(torch.float16)    # Mask: 1 channel
            targets = targets.to(device).to(torch.float16)  # Target image
            print("targets shape:", targets.shape)

            # Concatenate image and mask: 4 channels (3 image + 1 mask)
            combined_inputs = torch.cat([input_images, input_masks], dim=1)
            print("combined_inputs shape:", combined_inputs.shape)

            # Check input dimensions
            assert combined_inputs.shape[1] == 4, f"Expected 4 channels, got {combined_inputs.shape[1]}"

            # Generate random timesteps and hidden states
            batch_size = combined_inputs.shape[0]
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=input_images.device).long()
            encoder_hidden_states = pipe.text_encoder(
                pipe.tokenizer([""] * batch_size, return_tensors="pt", padding=True, truncation=True).input_ids.to(input_images.device)
            )[0]

            # Add noise to inputs
            noise = torch.randn_like(combined_inputs)  # Random noise
            noisy_inputs = combined_inputs + noise

            # Forward pass through the UNet
            outputs = pipe.unet(noisy_inputs, timesteps, encoder_hidden_states).sample

            # Compute loss
            loss = torch.nn.functional.mse_loss(outputs, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch Loss: {epoch_loss / len(dataloader)}")

    # Save LoRA weights
    print("Training complete. Saving LoRA weights...")
    pipe.unet.save_pretrained("./lora_weights")


# Main Function
def main():
    # Paths
    empty_rooms_dir = "./emptyRooms"
    manual_masks_dir = "./manualMasks"
    automatic_masks_dir = "./automaticMasks"
    masks_dir = automatic_masks_dir
    model_id = "stabilityai/stable-diffusion-2-inpainting"

    # Merge inputs and masks
    # merge_inputs_and_masks(empty_rooms_dir, masks_dir, inputs_dir)

    # Load dataset
    dataloader = load_dataset(inputs_dir=empty_rooms_dir, masks_dir=masks_dir, batch_size=4, img_size=512)

    # Setup model with LoRA
    pipe = setup_model_with_lora(model_id)

    # Train LoRA
    train_lora(pipe, dataloader)

if __name__ == "__main__":
    main()
