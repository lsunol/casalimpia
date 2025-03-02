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

DEBUG_MODE = False
# Función de ayuda para loguear mensajes de debug
def debug_log(message: str):
    """Imprime mensaje de debug si DEBUG_MODE está activo."""
    if DEBUG_MODE:
        logger.info(message)


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
                            num_inference_steps=30,
                            guidance_scale=args.guidance_scale,
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
               num_epochs=5, lr=1e-4, img_size=512, dtype="float32", 
               save_latent_representations=False, lora_rank=32, lora_alpha=16, lora_dropout=0.1,
               lora_target_modules=["to_k", "to_q", "to_v", "to_out.0"], overfitting=False, optimizer_type="AdamW", timestamp=None):
    
    # Determinar el tipo de dato
    torch_dtype = torch.float32 if dtype == "float32" else torch.float16
    debug_log(f"Base torch_dtype: {torch_dtype}")

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
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device, dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=torch_dtype)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device, dtype=torch.float32)
    debug_log(f"UNet forced dtype: {next(unet.parameters()).dtype}")
   
    # Congelar gradientes para VAE, text encoder y U-Net (se entrena solo LoRA)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    
    # Debug Mark 2: Check model weights dtype
    debug_log(f"UNet parameter dtype: {next(unet.parameters()).dtype}")
    debug_log(f"VAE parameter dtype: {next(vae.parameters()).dtype}")

    # Configurar LoRA en U-Net
    lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        init_lora_weights="gaussian",
        lora_dropout=lora_dropout,
    )

    if (overfitting):
        train_loader = sampling_loader
        wandb_run.tags = wandb_run.tags + ("OVERFITTING",)

    wandb.config.update({"num_images": len(train_loader.dataset)})

    # Agregar adaptadores LoRA
    unet.add_adapter(unet_lora_config)
    pipe.unet = unet

    # Parámetros LoRA
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))

    # Optimizador
    #optimizer = bnb.optim.Adam8bit(lora_layers, lr=lr)
    #optimizer = torch.optim.AdamW(lora_layers, lr=lr)
    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(lora_layers, lr=lr)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(lora_layers, lr=lr)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(lora_layers, lr=lr, momentum=0.9)
    elif optimizer_type == "RMSprop":
        optimizer = torch.optim.RMSprop(lora_layers, lr=lr)
    elif optimizer_type == "Adagrad":
        optimizer = torch.optim.Adagrad(lora_layers, lr=lr)
    logger.info(f"Using optimizer: {optimizer_type}, LR: {lr}")

    # Integrar Accelerate
    
    accelerator = Accelerator()
    unet, optimizer, train_loader = accelerator.prepare(
        unet, optimizer, train_loader
    )

    # GradScaler para mixed precision (si dtype == "float16")
    scaler = GradScaler()

    """
    # Scheduler ConstantLR (mantiene lr constante)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=1.0,  # Multiplicador del lr inicial (1.0 = sin cambio)
        total_iters=num_epochs * len(train_loader)  # Total de pasos (opcional, no afecta aquí)
    )
    
    # Scheduler OneCycleLR (LR sube hasta 1e-3 y luego decae)
    total_steps = num_epochs * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='linear'
    )
    """
    total_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * 0.1) if args.lr_scheduler in ["cosine", "linear"] else 0,
        num_training_steps=total_steps
    )
    debug_log(f"Scheduler: {args.lr_scheduler}, total_steps={total_steps}")
   
    # Noise scheduler para la difusión
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")    

    first_time_ever = True
    best_psnr = -float('inf')
    patience = 50  # Número de épocas sin mejora antes de parar deberia de ser 50 en el caso del dataset de 13k. 10 para ser muy optimistas (solo 1 fallo
    patience_counter = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        unet.train()
        train_epoch_loss = 0.0
        logger.info(f"Training on {len(train_loader.dataset)} images, batches per epoch: {len(train_loader)} with overfitting={overfitting}")

        for input_images, input_masks, targets, unpadded_masks in tqdm(train_loader):
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

                    loss = F.mse_loss(noise_pred, target_noise, reduction="mean")
                
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(lora_layers, max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logger.info(f"Batch loss: {loss.item()} | Learning rate: {lr_scheduler.get_last_lr()[0]}")
            wandb.log({"batch_loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0], "epoch": epoch + 1})

            train_epoch_loss += loss.item()

        # Promedio de pérdida por época
        #avg_epoch_loss = train_epoch_loss / len(train_loader)
        #wandb.log({"train_loss": avg_epoch_loss, "psnr": psnr, "epoch": epoch + 1})
        avg_train_loss = train_epoch_loss / len(train_loader)
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})
        
        # Validation phase
        unet.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for val_input_images, val_input_masks, val_targets, val_unpadded_masks in val_loader:
                val_input_images = val_input_images.to(device, dtype=torch_dtype)
                val_input_masks = val_input_masks.to(device, dtype=torch_dtype)
                val_targets = val_targets.to(device, dtype=torch_dtype)
                val_unpadded_masks = val_unpadded_masks.to(device, dtype=torch_dtype)

                val_target_latents = vae.encode(val_targets.to(torch_dtype)).latent_dist.sample() * vae.config.scaling_factor
                val_masked_latents = vae.encode(val_input_images.to(torch_dtype)).latent_dist.sample() * vae.config.scaling_factor

                val_mask = torch.stack([torch.nn.functional.interpolate(m.unsqueeze(0), size=(img_size // 8, img_size // 8)) for m in val_input_masks]).to(torch_dtype)
                val_mask = val_mask.reshape(-1, 1, img_size // 8, img_size // 8)

                val_noise = torch.randn_like(val_target_latents)
                val_bsz = val_target_latents.shape[0]
                val_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (val_bsz,), device=device).long()
                val_noisy_latents = noise_scheduler.add_noise(val_target_latents, val_noise, val_timesteps)

                val_latent_model_input = torch.cat([val_noisy_latents, val_mask, val_masked_latents], dim=1)
                val_encoder_hidden_states = text_encoder(pipe.tokenizer([EMPTY_ROOM_PROMPT[0]] * val_bsz, return_tensors="pt", padding=True, truncation=True).input_ids.to(device))[0].to(torch_dtype)

                with autocast(device_type="cuda", enabled=(dtype=="float16")):
                    val_noise_pred = unet(val_latent_model_input, val_timesteps, val_encoder_hidden_states).sample
                    if noise_scheduler.config.prediction_type == "epsilon":
                        val_target_noise = val_noise
                    val_loss = F.mse_loss(val_noise_pred, val_target_noise, reduction="mean")

                val_epoch_loss += val_loss.item()

        # Calculate average validation loss for the epoch
        avg_val_loss = val_epoch_loss / len(val_loader)
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})



        accelerator.print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        #accelerator.print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss}")

        #NEW ADD
        psnr = calculate_psnr_and_save_inpaint_samples(pipe, sampling_loader, epoch, train_dir)
        if psnr > best_psnr:
            best_psnr = psnr
            patience_counter = 0
            unet.save_attn_procs(f"{train_dir}_best_lora_weights")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                accelerator.print(f"Early stopping at epoch {epoch+1}")
                break

        unet.train()

        with open(metrics_log_file, "a") as f:
            metrics = {"epoch_loss": avg_train_loss, "val_loss": avg_val_loss, "psnr": psnr, "epoch": epoch + 1}
            wandb.log(metrics)
            logger.info(f"Epoch Loss: {metrics['epoch_loss']:.4f} | Val Loss: {metrics['val_loss']:.4f} | PSNR: {metrics['psnr']}")
            f.write(f"{epoch},{metrics['epoch_loss']:.4f},{metrics['val_loss']:.4f},{metrics['psnr']}\n")

        """
        # LR actual
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr, "epoch": epoch + 1})

        if (epoch + 1) % max(1, num_epochs // 10) == 0:
        #if (epoch + 1) % 10 == 0:
            unet.eval()
                    # Aquí usamos train_loader (o test_loader) para las muestras
            psnr = calculate_psnr_and_save_inpaint_samples(pipe, sampling_loader, epoch, train_dir)
            accelerator.print(f"Epoch {epoch+1} - Saved samples, PSNR: {psnr}")
            if psnr > best_psnr:
                best_psnr = psnr
                patience_counter = 0
                # Opcional: guardar mejor modelo
                unet.save_attn_procs(f"{train_dir}_best_lora_weights")
            else:
                patience_counter += 10
                if patience_counter >= patience:
                    accelerator.print(f"Early stopping at epoch {epoch+1}")
                    break
            unet.train()
            

        # Registrar métricas en CSV (ajusta PSNR si lo calculas)
        with open(metrics_log_file, "a") as f:
            metrics = {"epoch_loss": epoch_loss / len(train_loader), "psnr": psnr if 'psnr' in locals() else 0, "epoch": epoch + 1}
            wandb.log(metrics)
            logger.info(f"Epoch Loss: {metrics['epoch_loss']} | PSNR: {metrics['psnr']}")
            f.write(f"{epoch},{metrics['epoch_loss']},{metrics['psnr']}\n")
"""
    # Final del entrenamiento
    accelerator.wait_for_everyone()
    accelerator.print("Training complete. Saving final LoRA weights...")
    unet = accelerator.unwrap_model(unet)
    #assert hasattr(unet, "peft_config"), "El modelo UNet no tiene configurado LoRA."
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
    parser.add_argument("--initial-learning-rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--lr-scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'),)
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="Dropout probability for LoRA layers (default: 0.1)")
    parser.add_argument("--lora-target-modules", type=str, nargs='+', default=["to_k", "to_q", "to_v", "to_out.0"], help="Target modules for LoRA adaptation")
    parser.add_argument("--guidance-scale", type=float, default=1, help="Guidance scale for inpainting")
    parser.add_argument("--overfitting", action='store_true', help="Enable overfitting mode")
    parser.add_argument("--no-overfitting", dest="overfitting", action='store_false', help="Disable overfitting mode")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adam", "SGD", "RMSprop", "Adagrad"], help="Optimizer algorithm")
    parser.set_defaults(overfitting=False)
    
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
wandb_run = wandb.init(project="casalimpia", tags=[])

# Main Function
def main():
    
    initial_timestamp = datetime.now()

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
        "epochs": args.epochs, 
        "num_epochs": args.epochs, 
        "batch_size": args.batch_size,
        "initial_lr": args.initial_learning_rate, 
        "img_size": args.img_size, 
        "dtype": args.dtype, 
        "lora_rank": args.lora_rank, 
        "lora_alpha": args.lora_alpha,
        "l_rank": args.lora_rank, 
        "l_alpha": args.lora_alpha,
        "l_dropout": args.lora_dropout,
        "l_modules": args.lora_target_modules,
        "guidance_scale": args.guidance_scale,
        "lr_scheduler": args.lr_scheduler,
        "overfitting": args.overfitting,
        "optimizer": args.optimizer,})

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
               lora_dropout=args.lora_dropout,
               lora_target_modules=args.lora_target_modules,
               dtype=args.dtype,
               overfitting=args.overfitting,
               optimizer_type=args.optimizer,
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
