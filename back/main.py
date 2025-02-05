import torch
import torchvision.transforms as transforms
from mit_semseg.utils import colorEncode
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from diffusers import AutoencoderKL
from safetensors.torch import save_file
import argparse
from datetime import datetime
from empty_rooms_dataset import load_dataset
import os
from image_service import create_epoch_image

# Select GPU if available
if not torch.cuda.is_available():
    print("No GPU available. This script requires GPU to run.")
    exit(1)
device = torch.device("cuda")

# Enable flash attention, memory-efficient, and math optimizations if available
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
EMPTY_ROOM_PROMPT = "A photo of an empty room with bare walls, clean floor, and no furniture or objects."

# Setup LoRA Model
def setup_model_with_lora(model_id):

    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    # Add autoencoder to the pipeline
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    vae.requires_grad_(False) 
    pipe.vae = vae.to(torch.float32)

    # Inspect available modules in the UNet
    supported_modules = []

    for name, module in pipe.unet.named_modules():
        # Check if module is compatible with LoRA
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):                              
            supported_modules.append(name)

    # Ensure there are supported modules
    if not supported_modules:                                                                   
        raise ValueError("No compatible modules found for LoRA in the UNet.")

    # Setup LoRA with compatible modules
    lora_config = LoraConfig(                                                                       
        r=16,
        lora_alpha=32,
        target_modules=supported_modules,
        lora_dropout=0.1
    )

    pipe.unet = get_peft_model(pipe.unet, lora_config)

    return pipe

# Add this new function after setup_model_with_lora and before train_lora
def save_inpaint_samples(pipe, dataloader, epoch, output_dir):
    """Generate and save inpainted samples after each epoch"""

    try:
        # Store original state
        pipe.unet.eval()

        with torch.no_grad():
            # Get a batch of images
            input_images, input_masks, target_images = next(iter(dataloader))

            # Process on GPU
            input_images = input_images.to(device)
            input_masks = input_masks.to(device)

            # Denormalize images
            input_images = ((input_images + 1) / 2).clamp(0, 1)
            target_images = ((target_images + 1) / 2).clamp(0, 1)

            max_samples = 3
            # Only process up to 3 images from the batch
            for idx in range(min(max_samples, input_images.size(0))):
                # Convert input images to PIL format
                pil_img = transforms.ToPILImage()(input_images[idx].cpu())
                pil_mask = transforms.ToPILImage()(input_masks[idx].cpu())
                pil_target = transforms.ToPILImage()(target_images[idx].cpu())

                # Convert to float16 for inference
                with torch.autocast(device.type):
                    inferred_image = pipe(
                        image=pil_img,
                        mask_image=pil_mask,
                        prompt=EMPTY_ROOM_PROMPT,
                        num_inference_steps=20,
                    ).images

                create_epoch_image(input_image=pil_img, 
                                input_mask=pil_mask,
                                inferred_image=inferred_image[0], 
                                target_image=pil_target,
                                epoch=epoch, 
                                sample_index=idx,
                                output_path=output_dir)

    except Exception as e:
        print(f"Error during sample generation: {str(e)}")

    finally:
        # Ensure we return to training mode
        pipe.unet.train()

# Training LoRA: concatenates images and masks into a single tensor for training (6 chanel input)
def train_lora(pipe, train_loader, test_loader, val_loader, output_dir="back/data", num_epochs=5, lr=5e-5):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir = f"{output_dir}/lora_trains/{timestamp}/"
    os.makedirs(train_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        pipe.unet.train()
        epoch_loss = 0.0

        for input_images, input_masks, targets in tqdm(train_loader):

            # Move to device
            input_images = input_images.to(device).to(torch.float16)
            input_masks = input_masks.to(device).to(torch.float16)
            targets = targets.to(device).to(torch.float16)

            # Codificar imágenes en el espacio latente
            # latents = pipe.vae.encode(input_images).latent_dist.sample() * pipe.vae.config.scaling_factor
            latents = pipe.vae.encode(input_images.to(torch.float32)).latent_dist.sample() * pipe.vae.config.scaling_factor

            # Codificar imágenes objetivo (targets) en el espacio latente
            # target_latents = pipe.vae.encode(targets).latent_dist.sample() * pipe.vae.config.scaling_factor
            target_latents = pipe.vae.encode(targets.to(torch.float32)).latent_dist.sample() * pipe.vae.config.scaling_factor

            # Redimensionar las máscaras al tamaño de los latentes
            masks_resized = torch.nn.functional.interpolate(input_masks, size=latents.shape[-2:])
            
            # Concatenar latentes y máscaras. Asegurarse de que haya 9 canales:
            # latentes (4), máscaras (1), y otra copia de latentes (4) => Total = 9 canales
            # TODO: cambiar los 4 últimos latentes por los latents de combinación de imagen + máscara
            combined_inputs = torch.cat([latents, masks_resized, latents], dim=1)

            # Generar ruido y añadirlo a los latentes
            batch_size = combined_inputs.shape[0]
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(combined_inputs)
            noisy_inputs = combined_inputs + noise  # Añadimos el ruido

            # Codificación del prompt vacío (sin texto)
            encoder_hidden_states = pipe.text_encoder(
                pipe.tokenizer([EMPTY_ROOM_PROMPT] * batch_size, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            )[0].to(torch.float16)

            noisy_inputs = noisy_inputs.to(torch.float16)
            encoder_hidden_states = encoder_hidden_states.to(torch.float16)

            # Forward pass through the UNet
            outputs = pipe.unet(noisy_inputs, timesteps, encoder_hidden_states).sample
            outputs = outputs.to(torch.float16)
            target_latents = target_latents.to(torch.float16)

            # Compute loss
            loss = torch.nn.functional.mse_loss(outputs, target_latents)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        pipe.unet.eval()
        save_inpaint_samples(pipe, test_loader, epoch, train_dir)
        pipe.unet.train()

        print(f"Epoch Loss: {epoch_loss / len(train_loader)}")

    # Save LoRA weights
    print("Training complete. Saving LoRA weights...")
    num_images = len(train_loader.dataset)
    pipe.unet.save_pretrained(f"{train_dir}_lora_weights_{num_epochs}_epochs_{num_images}_images")



def read_parameters():

    parser = argparse.ArgumentParser(description="Entrenar modelo de segmentación con LoRA")
    parser.add_argument("--empty-rooms-dir", type=str, required=True, help="Dataset folder containing images of empty rooms")
    parser.add_argument("--masks-dir", type=str, required=True, help="Dataset folder containing images of masks")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--output-dir", type=str, default="./data/trained_lora", help="Output directory for saving LoRA weights")

    args = parser.parse_args()

    return args.empty_rooms_dir, args.masks_dir, args.epochs, args.batch_size, args.output_dir

# Main Function
def main():
    
    initial_timestamp = datetime.now()

    empty_rooms_dir, masks_dir, epochs, batch_size, output_dir = read_parameters()

    train_loader, val_loader, test_loader = load_dataset(
        inputs_dir=empty_rooms_dir, 
        masks_dir=masks_dir, 
        batch_size=batch_size, 
        img_size=512,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42)

    pipe = setup_model_with_lora(MODEL_ID)

    train_lora(pipe, train_loader, test_loader, val_loader, num_epochs=epochs, output_dir=output_dir)

    final_timestamp = datetime.now()
    print(f"Training completed. Initial timestamp: {initial_timestamp.strftime(TIMESTAMP_FORMAT)}.")
    print(f"Final timestamp: {final_timestamp.strftime(TIMESTAMP_FORMAT)}.")
    elapsed_time = final_timestamp - initial_timestamp
    print(f"Elapsed time in seconds: {elapsed_time.total_seconds()}.")

if __name__ == "__main__":
    main()
