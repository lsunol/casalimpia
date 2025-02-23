# Librerías estándar de Python
import argparse
import os
import json
import logging
import wandb
from datetime import datetime
from image_service import save_image

from accelerate import Accelerator
from diffusers.optimization import get_scheduler


# Manipulación de imágenes y datos
import numpy as np


# PyTorch y torchvision
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms

# Hugging Face y Diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_xformers_available

from diffusers.optimization import get_scheduler  # Optimizadores personalizados
from diffusers.utils import check_min_version, is_xformers_available  # Verificación de versiones y optimización de memoria


# Transformers (Hugging Face)
from transformers import CLIPTextModel

# PEFT (Parameter-Efficient Fine-Tuning)
from peft import LoraConfig

# Hugging Face Hub
from huggingface_hub import create_repo, upload_folder  # Funcionalidades para subir modelos al Hub

# Progreso y logging
from tqdm import tqdm


# Módulos personalizados
from empty_rooms_dataset import load_dataset  # Carga del dataset de habitaciones vacías
from image_service import save_epoch_sample  # Servicio para guardar ejemplos durante el entrenamiento

from torch.amp import GradScaler, autocast
from metrics import calculate_psnr
import bitsandbytes as bnb

# Select GPU if available
if not torch.cuda.is_available():
    print("No GPU available. This script requires GPU to run.")
    exit(1)
device = torch.device("cuda")

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    print("Flash attention and related optimizations are ENABLED.")
else:
    print("CUDA is not available; flash attention optimizations are DISABLED.")

# Enable flash attention, memory-efficient, and math optimizations if available. Optimizaciónes en CUDA
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

# Modelos predefinidos
MODELS = {
    "stability-ai": "stabilityai/stable-diffusion-2-inpainting", 
    "runway": "runwayml/stable-diffusion-inpainting"
    }
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"              # Formato de timestamp(marca de timepo) para guardado de datos
EMPTY_ROOM_PROMPT = [                           
    "A photo of an empty room with bare walls, clean floor, and no furniture or objects.",
    "Remove everything from the room except the walls, windows, doors and floor.",
    "Fill in the missing areas with structural background only, preserving the room's geometry. Do not generate furniture, decorations, or any identifiable objects. Maintain a uniform surface for walls, floors, and ceilings, blending seamlessly with the existing structure."
]

# Generación de ejemplos después de cada época de entrenamiento
# Add this new function after setup_model_with_lora and before train_lora
def calculate_psnr_and_save_inpaint_samples(pipe, dataloader, epoch, output_dir):
    """Generate and save inpainted samples after each epoch"""

    try:
        # Guarda el estado original: se pone el U-Net en modo evaluación
        pipe.unet.eval()        # Cambia a modo evaluación pare el UNET para que no se actualizen los pesos

        with torch.no_grad():

            psnr_values = []

            for input_images, input_masks, target_images, _ in dataloader:

                # Process on GPU
                input_images = input_images.to(device)
                input_masks = input_masks.to(device)

                # Denormalize images
                input_images = ((input_images + 1) / 2).clamp(0, 1)
                target_images = ((target_images + 1) / 2).clamp(0, 1)

                for idx in range(input_images.size(0)):

                    with torch.autocast(device.type):
                        inferred_image = pipe(
                            image=input_images[idx],
                            mask_image=input_masks[idx],
                            prompt=EMPTY_ROOM_PROMPT,
                            num_inference_steps=20,
                        ).images

                    current_psnr = calculate_psnr(inferred_image, target_images[idx])
                    psnr_values.append(current_psnr)
                        
                    # Convert input images to PIL format
                    pil_img = transforms.ToPILImage()(input_images[idx].cpu())
                    pil_mask = transforms.ToPILImage()(input_masks[idx].cpu())
                    pil_target = transforms.ToPILImage()(target_images[idx].cpu())

                    images_to_log = [
                        wandb.Image(pil_img, caption=f"input - epoch: {epoch + 1}"),
                        wandb.Image(pil_target, caption=f"target - epoch: {epoch + 1}"),
                        wandb.Image(inferred_image[0], caption=f"inferred - epoch: {epoch + 1}")
                    ]
                    wandb.log({"images": images_to_log, "epoch": epoch + 1})

                    save_epoch_sample(input_image=pil_img, 
                                    input_mask=pil_mask,
                                    inferred_image=inferred_image[0], 
                                    target_image=pil_target,
                                    epoch=epoch, 
                                    sample_index=idx,
                                    output_path=output_dir)

            return np.mean(psnr_values)

    except Exception as e:
        print(f"Error during sample generation: {str(e)}")  # Captura y muestra cualquier error durante la generación de muestras.

    finally:
        pipe.unet.train()        # Vuelve a modo entrenamiento


# Training LoRA: concatenates images and masks into a single tensor for training (6 chanel input)
def train_lora(model_id, train_loader, test_loader, val_loader, sampling_loader, train_dir, 
               num_epochs=5, lr=1e-5, img_size=512, dtype="float32", 
               save_latent_representations=False, lora_rank=16, lora_alpha=32, timestamp=None):

    # Determinar el tipo de dato
    torch_dtype = torch.float32 if dtype == "float32" else torch.float16
    num_images = len(train_loader.dataset)

    # Archivo para guardar métricas de entrenamiento
    metrics_log_file = os.path.join(train_dir, "training_metrics.csv")
    with open(metrics_log_file, "w") as f:
        f.write("epoch,epoch_loss,avg_psnr\n")

    # Cargar el pipeline de inpainting
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype, safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Cargar componentes: text encoder, VAE y U-Net
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Congelar gradientes para VAE, text encoder y U-Net (se entrena solo LoRA)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Enviar modelos a GPU y configurar tipo de dato
    unet.to(device, dtype=torch_dtype)
    vae.to(device, dtype=torch_dtype)
    text_encoder.to(device, dtype=torch_dtype)

    # Configurar LoRA en U-Net
    lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    lora_dropout = 0.1
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        init_lora_weights="gaussian",
        lora_dropout=lora_dropout,
    )

    # Guardar config usada
    config_used = {
        "model_id": model_id,
        "num_epochs": num_epochs,
        "num_images": num_images,
        "lr_scheduler": args.lr_scheduler,
        "lr": lr,
        "img_size": img_size,
        "dtype": dtype,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target_modules": lora_target_modules,
        "lora_dropout": lora_dropout,
        "save_latent_representations": save_latent_representations,
        "timestamp": timestamp
    }
    with open(os.path.join(train_dir, "config_used.json"), "w") as file:
        json.dump(config_used, file, indent=4)

    # Agregar adaptadores LoRA
    unet.add_adapter(unet_lora_config)
    pipe.unet = unet

    # Parámetros LoRA
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))

    # Optimizador
    #optimizer = bnb.optim.Adam8bit(lora_layers, lr=lr)
    optimizer = torch.optim.AdamW(lora_layers, lr=lr)


        # Integrar Accelerate
    from accelerate import Accelerator
    accelerator = Accelerator()
    unet, optimizer, train_loader = accelerator.prepare(
        unet, optimizer, train_loader
    )

    # GradScaler para mixed precision (si dtype == "float16")
    scaler = GradScaler()

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(len(train_loader) * num_epochs * 0.1),  # 10% of total steps as warmup
        num_training_steps=len(train_loader) * num_epochs,
        # num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
	
    # Noise scheduler para la difusión
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")    

    first_time_ever = True
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        unet.train()
        epoch_loss = 0.0

        for input_images, input_masks, targets, unpadded_masks in tqdm(sampling_loader):

            # Move to device
            input_images = input_images.to(device, dtype=torch_dtype)
            input_masks = input_masks.to(device, dtype=torch_dtype)
            targets = targets.to(device, dtype=torch_dtype)
            unpadded_masks = unpadded_masks.to(device, dtype=torch_dtype)

            # Codificar con VAE
            # https://github.com/huggingface/diffusers/blob/main/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint_lora.py

            # originaly named "latents" in train_dreambooth_inpaint_lora.py, here I used "target_latents" to make it more clear
            target_latents = vae.encode(targets.to(torch_dtype)).latent_dist.sample() * vae.config.scaling_factor
            masked_latents = vae.encode(input_images.to(torch_dtype)).latent_dist.sample() * vae.config.scaling_factor         

            # Redimensionar la máscara a la resolución latente
            mask = torch.stack([
                torch.nn.functional.interpolate(m.unsqueeze(0), size=(img_size // 8, img_size // 8))
                for m in input_masks
            ]).to(torch_dtype)
            mask = mask.reshape(-1, 1, img_size // 8, img_size // 8)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(target_latents)
            bsz = target_latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

            # Concatenar: [B, 9, 64, 64]
            latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

            # Condicionamiento textual
            encoder_hidden_states = text_encoder(
                pipe.tokenizer([EMPTY_ROOM_PROMPT[0]] * bsz, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            )[0].to(torch_dtype)

            if first_time_ever:
                psnr = calculate_psnr_and_save_inpaint_samples(pipe, sampling_loader, -1, train_dir)
                logger.info(f"Initial PSNR: {psnr}")

                first_time_ever = False

                if save_latent_representations:
                    to_pil = transforms.ToPILImage()
                    normalized_input_images = (input_images / 2 + 0.5).clamp(0, 1)
                    normalized_targets = (targets / 2 + 0.5).clamp(0, 1)
                    normalized_masks = (input_masks / 2 + 0.5).clamp(0, 1)
                    normalized_unpadded_masks = (unpadded_masks / 2 + 0.5).clamp(0, 1)
                    normalized_masks = normalized_masks.repeat(1, 3, 1, 1)  # Convert from 1 to 3 channels
                    normalized_unpadded_masks = normalized_unpadded_masks.repeat(1, 3, 1, 1)  # Convert from 1 to 3 channels

                    decoded_target_latents = vae.decode(target_latents / vae.config.scaling_factor).sample
                    decoded_target_latents = (decoded_target_latents / 2 + 0.5).clamp(0, 1)
                    decoded_masked_latents = vae.decode(masked_latents / vae.config.scaling_factor).sample
                    decoded_masked_latents = (decoded_masked_latents / 2 + 0.5).clamp(0, 1)

                    for i in range(decoded_target_latents.shape[0]):
                        original_image_diff = torch.abs(normalized_targets[i] - decoded_target_latents[i])
                        original_image_comparison = (torch.cat((normalized_targets[i], original_image_diff, decoded_target_latents[i]), dim=2))

                        masked_image_diff = torch.abs(normalized_input_images[i] - decoded_masked_latents[i])
                        masked_image_comparison = (torch.cat((normalized_input_images[i], masked_image_diff, decoded_masked_latents[i]), dim=2))                        

                        mask_diff = torch.abs(normalized_unpadded_masks[i] - normalized_masks[i])
                        unpadded_padded_mask_comparison = (torch.cat((normalized_unpadded_masks[i], mask_diff, normalized_masks[i]), dim=2))

                        final_img = to_pil(torch.cat((original_image_comparison, masked_image_comparison, unpadded_padded_mask_comparison), dim=1))
                        final_img.save(f"{train_dir}sample_{i}_decoded_target_latents.png")
            
			# Entrenamiento con acumulación de gradientes
            with accelerator.accumulate(unet):
                with autocast(device_type="cuda", enabled=(dtype=="float16")):
                    noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                    # Determinar el target según el tipo de predicción
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target_noise = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target_noise = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(noise_pred.float(), target_noise.float(), reduction="mean")

            		logger.info(f"Batch loss: {loss.item()} | Learning rate: {lr_scheduler.get_last_lr()[0]}")
            		wandb.log({"batch_loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0], "epoch": epoch + 1})
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(lora_layers, max_norm=1.0)
                optimizer.step()
            	lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            wandb.log({"epoch_loss": epoch_loss / len(train_loader), "psnr": psnr, "epoch": epoch + 1})

        # Promedio de pérdida por época
        avg_epoch_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch_loss": avg_epoch_loss})
        accelerator.print(f"Epoch {epoch+1} Loss: {avg_epoch_loss}")

        # LR actual
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr})

        # Llamar a la función de sample cada 10 épocas
        unet.eval()
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            # Aquí usamos train_loader (o test_loader) para las muestras
            psnr = calculate_psnr_and_save_inpaint_samples(pipe, sampling_loader, epoch, train_dir)
            wandb.log({"sample_psnr": psnr})
            accelerator.print(f"Epoch {epoch+1} - Saved samples, PSNR: {psnr}")
        unet.train()

        # Registrar métricas en CSV (ajusta PSNR si lo calculas)
        with open(metrics_log_file, "a") as f:
		logger.info(f"Epoch Loss: {epoch_loss / len(train_loader)} | PSNR: {psnr}")
            f.write(f"{epoch},{avg_epoch_loss},{psnr if 'psnr' in locals() else 0}\n")

    # Final del entrenamiento
    accelerator.wait_for_everyone()
    accelerator.print("Training complete. Saving final LoRA weights...")
    unet = accelerator.unwrap_model(unet)
    assert hasattr(unet, "peft_config"), "El modelo UNet no tiene configurado LoRA."
    unet.save_attn_procs(f"{train_dir}_lora_weights", unet_lora_layers=pipe.unet.attn_processors)


# Función para leer y parsear los parámetros pasados por línea de comando
def read_parameters():

    parser = argparse.ArgumentParser(description="Entrenar modelo de segmentación con LoRA")
    parser.add_argument("--empty-rooms-dir", type=str, required=True, help="Dataset folder containing images of empty rooms")
    parser.add_argument("--masks-dir", type=str, required=True, help="Dataset folder containing images of masks")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--output-dir", type=str, default="./data/trained_lora", help="Output directory for saving LoRA weights")
    parser.add_argument("--model", type=str, choices=["stability-ai", "runway"], default="stability-ai", help="Model to use: \"stability-ai\" (default) or \"runway\"")
    parser.add_argument("--img-size", type=int, default=512, help="Image size for training")
    parser.add_argument("--save-latent-representations", action="store_true", help="Save latent representations during training")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float32", help="Data type for training: float16 or float32")
    parser.add_argument("--lora-rank", type=int, default=64, help="Rank for LoRA layers")
    parser.add_argument("--lora-alpha", type=int, default=128, help="Alpha scaling factor for LoRA layers")
    parser.add_argument("--initial-learning-rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--lr-scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'),)

    args = parser.parse_args()

    return args

args = read_parameters()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train_dir = f"{args.output_dir}/lora_trains/{timestamp}_{args.epochs}_epochs/"
os.makedirs(train_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{train_dir}train.log", mode="a")],
    )

logger = logging.getLogger()

# Main Function
def main():
    
    initial_timestamp = datetime.now()

    wandb.init(project="casalimpia")

    logger.info("Start training")

    train_loader, val_loader, test_loader, sampling_loader = load_dataset(
        inputs_dir=args.empty_rooms_dir, 
        masks_dir=args.masks_dir, 
        batch_size=args.batch_size, 
        mask_padding=0,
        img_size=args.img_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        logger=logger)

    model_id = MODELS[args.model]

    wandb.run.name = f"{args.epochs:03d}_epochs_{len(train_loader.dataset):04d}_images_{args.lora_rank:03d}_rank_{args.lora_alpha:03d}_alpha"
    wandb.config.update({
        "model": model_id,
        "num_epochs": args.epochs, 
        "batch_size": args.batch_size,
        "num_images": len(train_loader.dataset), 
        "initial_lr": args.initial_learning_rate, 
        "img_size": args.img_size, 
        "dtype": args.dtype, 
        "lora_rank": args.lora_rank, 
        "lora_alpha": args.lora_alpha})

    train_lora(model_id, 
               train_loader, 
               test_loader, 
               val_loader, 
               sampling_loader,
               num_epochs=args.epochs,
               lr=args.initial_learning_rate, 
               train_dir=train_dir,
               img_size=args.img_size, 
               save_latent_representations=args.save_latent_representations,
               lora_rank=args.lora_rank,
               lora_alpha=args.lora_alpha,
               dtype=args.dtype,
               timestamp=timestamp)

    final_timestamp = datetime.now()
    logger.info(f"Training completed. Initial timestamp: {initial_timestamp.strftime(TIMESTAMP_FORMAT)}.")
    logger.info(f"Final timestamp: {final_timestamp.strftime(TIMESTAMP_FORMAT)}.")
    elapsed_time = final_timestamp - initial_timestamp
    logger.info(f"Elapsed time in seconds: {elapsed_time.total_seconds()}.")
    logger.info("End training")
    wandb.config.update({
        "elapsed_time": elapsed_time.total_seconds(),
        "status": "completed"})

if __name__ == "__main__":
    main()
