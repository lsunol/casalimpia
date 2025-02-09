# Librerías estándar de Python
import argparse
import os
from datetime import datetime
import hashlib as insecure_hashlib  # Renombrado para evitar conflictos

# Manipulación de imágenes y datos
import numpy as np
from PIL import Image, ImageDraw

# PyTorch y torchvision
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

# Hugging Face y Diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_xformers_available
from diffusers.loaders import AttnProcsLayers

# Transformers (Hugging Face)
from transformers import CLIPTextModel, CLIPTokenizer

# PEFT (Parameter-Efficient Fine-Tuning)
from peft import LoraConfig, get_peft_model

# Accelerate (Hugging Face)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# Hugging Face Hub
from huggingface_hub import create_repo, upload_folder

# Progreso y logging
from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm

# Módulos personalizados
from empty_rooms_dataset import load_dataset
from image_service import create_epoch_image

# Otras utilidades
from mit_semseg.utils import colorEncode
from safetensors.torch import save_file

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

MODELS = {
    "stability-ai": "stabilityai/stable-diffusion-2-inpainting", 
    "runway": "runwayml/stable-diffusion-inpainting"
    }
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
EMPTY_ROOM_PROMPT = "A photo of an empty room with bare walls, clean floor, and no furniture or objects."

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
def train_lora(model_id, train_loader, test_loader, val_loader, output_dir="back/data", num_epochs=5, lr=5e-5, img_size=512):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_images = len(train_loader.dataset)
    train_dir = f"{output_dir}/lora_trains/{timestamp}_{num_epochs}_epochs_{num_images}_images/"
    os.makedirs(train_dir, exist_ok=True)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Add autoencoder to the pipeline
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Only train additional adapter LoRA layers
    vae.requires_grad_(False) 
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    
    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
        lora_dropout=0.1,
    )

    unet.add_adapter(unet_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=5e-5
    )

    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")    

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        unet.train()
        epoch_loss = 0.0

        for input_images, input_masks, targets in tqdm(train_loader):

            # Move to device
            input_images = input_images.to(device).to(torch.float16)
            input_masks = input_masks.to(device).to(torch.float16)
            targets = targets.to(device).to(torch.float16)

            # Codificar imágenes en el espacio latente
            # latents = pipe.vae.encode(input_images).latent_dist.sample() * pipe.vae.config.scaling_factor
            latents = vae.encode(targets.to(torch.float16)).latent_dist.sample() * pipe.vae.config.scaling_factor

            masked_latents = vae.encode(input_images).latent_dist.sample() * pipe.vae.config.scaling_factor

            mask = torch.stack(
                [torch.nn.functional.interpolate(mask.unsqueeze(0), size=(img_size // 8, img_size // 8)) for mask in input_masks]
            ).to(torch.float16)
            mask = mask.reshape(-1, 1, img_size // 8, img_size // 8)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # concatenate the noised latents with the mask and the masked latents
            latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

            # TODO: What's this?, the prompt?
            # Get the text embedding for conditioning
            # encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Codificación del prompt 
            encoder_hidden_states = pipe.text_encoder(
                pipe.tokenizer([EMPTY_ROOM_PROMPT] * bsz, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            )[0].to(torch.float16)

            # Predict the noise residual
            noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        pipe.unet.eval()
        save_inpaint_samples(pipe, test_loader, epoch, train_dir)
        pipe.unet.train()

        print(f"Epoch Loss: {epoch_loss / len(train_loader)}")

    # Save LoRA weights
    print("Training complete. Saving LoRA weights...")
    pipe.unet.save_pretrained(f"{train_dir}_lora_weights")



def read_parameters():

    parser = argparse.ArgumentParser(description="Entrenar modelo de segmentación con LoRA")
    parser.add_argument("--empty-rooms-dir", type=str, required=True, help="Dataset folder containing images of empty rooms")
    parser.add_argument("--masks-dir", type=str, required=True, help="Dataset folder containing images of masks")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--output-dir", type=str, default="./data/trained_lora", help="Output directory for saving LoRA weights")
    parser.add_argument("--model", type=str, choices=["stability-ai", "runway"], default="stability-ai", help="Model to use: \"stability-ai\" (default) or \"runway\"")
    parser.add_argument("--img-size", type=int, default=512, help="Image size for training")

    args = parser.parse_args()

    return args.empty_rooms_dir, args.masks_dir, args.epochs, args.batch_size, args.output_dir, args.model, args.img_size

# Main Function
def main():
    
    initial_timestamp = datetime.now()

    empty_rooms_dir, masks_dir, epochs, batch_size, output_dir, model_id_parameter, img_size = read_parameters()

    train_loader, val_loader, test_loader = load_dataset(
        inputs_dir=empty_rooms_dir, 
        masks_dir=masks_dir, 
        batch_size=batch_size, 
        img_size=img_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42)

    model_id = MODELS[model_id_parameter]

    train_lora(model_id, train_loader, test_loader, val_loader, num_epochs=epochs, output_dir=output_dir, img_size=img_size)

    final_timestamp = datetime.now()
    print(f"Training completed. Initial timestamp: {initial_timestamp.strftime(TIMESTAMP_FORMAT)}.")
    print(f"Final timestamp: {final_timestamp.strftime(TIMESTAMP_FORMAT)}.")
    elapsed_time = final_timestamp - initial_timestamp
    print(f"Elapsed time in seconds: {elapsed_time.total_seconds()}.")

if __name__ == "__main__":
    main()
