# Librerías estándar de Python
import argparse
import os
import json
import logging
import wandb
from wandb import Artifact
import time

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
from metrics import calculate_psnr, calculate_ssim, calculate_lpips

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

def upload_safetensors_to_wandb(file_path, artifact_name, artifact_type="model"):
    artifact = Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)


def infer_and_calculate_metrics(masked_image, mask, original_image, label, epoch, pipe, output_dir, 
                                calculate_metrics=True):
    # Process on GPU
    masked_image = masked_image.to(device)
    mask = mask.to(device)

    # Denormalize images
    masked_image = ((masked_image + 1) / 2).clamp(0, 1)
    original_image = ((original_image + 1) / 2).clamp(0, 1)

    with torch.autocast(device.type):
        inferred_test_images = pipe(
            image=masked_image,
            mask_image=mask,
            prompt=EMPTY_ROOM_PROMPT[0],
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
        ).images

    # Convert input images to PIL format
    to_pil = transforms.ToPILImage()  # Create the transformer
    pil_img = to_pil(masked_image.cpu())  # Convert single image tensor to PIL
    pil_mask = to_pil(mask.cpu())  # Convert single mask tensor to PIL 
    pil_target = to_pil(original_image.cpu())  # Convert single target tensor to PIL

    image_groups = {
        "train": 1,
        "test": 2,
        "furnished": 3
    }

    images_to_log = [
        wandb.Image(pil_img, caption=f"{label} input - epoch: {epoch + 1}", grouping=image_groups[label]),
        wandb.Image(pil_target, caption=f"{label} target - epoch: {epoch + 1}", grouping=image_groups[label]),
    ]

    psnr_values = []
    ssim_values = []
    lpips_values = []

    # Convert tensor outputs to numpy arrays for metric calculation
    for i in range(len(inferred_test_images)):

        inferred_test_image = np.array(inferred_test_images[i])

        images_to_log.extend([
            wandb.Image(inferred_test_image, caption=f"{label} inferred ({i}) - epoch: {epoch + 1}", grouping=image_groups[label])
        ])

        if args.save_epoch_result_images:
            save_epoch_sample(input_image=pil_img, 
                            input_mask=pil_mask,
                            inferred_image=inferred_test_image, 
                            target_image=pil_target,
                            epoch=epoch, 
                            sample_index=1,
                            output_path=output_dir)

        if (calculate_metrics):
            psnr = calculate_psnr(inferred_test_image, original_image)
            psnr_values.append(psnr)
            ssim = calculate_ssim(inferred_test_image, original_image)
            ssim_values.append(ssim)
            lpips = calculate_lpips(inferred_test_image, original_image, device)
            lpips_values.append(lpips)

    if (calculate_metrics):
        mean_psnr = np.mean(psnr_values)
        mean_ssim = np.mean(ssim_values)
        mean_lpips = np.mean(lpips_values)

        logger.info(f"{label} PSNR: {mean_psnr} | {label} SSIM: {mean_ssim}, {label} LPIPS: {mean_lpips}")
        wandb.log({f"{label}_psnr": mean_psnr, f"{label}_ssim": mean_ssim, f"{label}_lpips": mean_lpips, "epoch": epoch + 1})

    return images_to_log
    # wandb.log({"images": images_to_log, "type": label, "epoch": epoch + 1})

# Generación de ejemplos después de cada época de entrenamiento
# Add this new function after setup_model_with_lora and before train_lora
def evaluate_and_save_samples(pipe, train_set, test_set, rooms_with_furniture_loader, epoch, output_dir):
    """
    Generate and save inpainted samples after each epoch
    
    Args:
        pipe: StableDiffusionInpaintPipeline
        train_loader: DataLoader for training images
        sampling_loader: DataLoader for sampling images
        epoch: Current epoch number
        output_dir: Directory to save samples
        max_num_train_psnr_images: Maximum number of training images to use for PSNR calculation
    """

    try:
        pipe.unet.eval()

        with torch.no_grad():

            # Guarda el estado original: se pone el U-Net en modo evaluación
            pipe.unet.eval()        # Cambia a modo evaluación pare el UNET para que no se actualizen los pesos

            with torch.no_grad():

                for i in range(len(train_set[0])):

                    # Use stored tuple instead of iterating through dataloader
                    train_image = train_set[0][i]
                    train_mask = train_set[1][i]
                    train_target = train_set[2][i]

                    images_to_log = infer_and_calculate_metrics(train_image, train_mask, train_target, "train", 
                                                epoch, pipe, output_dir)

                for i in range(len(test_set[0])):

                    # Use stored tuple instead of iterating through dataloader
                    test_image = test_set[0][i]
                    test_mask = test_set[1][i]
                    test_target = test_set[2][i]

                    images_to_log.extend(
                        infer_and_calculate_metrics(test_image, test_mask, test_target, "test", epoch, pipe, output_dir))

                for i in range(len(rooms_with_furniture_loader.dataset)):

                    # Get image from rooms_with_furniture_loader
                    furnished_batch = next(iter(rooms_with_furniture_loader))
                    furnished_image = furnished_batch[0][0]
                    furniture_mask = furnished_batch[1][0]

                    images_to_log.extend(
                        infer_and_calculate_metrics(furnished_image, furniture_mask, furnished_image, "furnished", epoch, pipe, output_dir, calculate_metrics = False))

            wandb.log({"images": images_to_log, "epoch": epoch + 1})
    finally:
        pipe.unet.train()


# Training LoRA: concatenates images and masks into a single tensor for training (6 chanel input)
def train_lora(model_id, train_loader, test_loader, val_loader, sampling_loader, overfitting_loader, rooms_with_furniture_loader, train_dir, 
               num_epochs=5, lr=1e-4, img_size=512, dtype="float32", 
               save_latent_representations=False, save_epoch_tensors=False, lora_rank=32, lora_alpha=16, lora_dropout=0.1,
               lora_target_modules=["to_k", "to_q", "to_v", "to_out.0"], overfitting=False, timestamp=None):
    
    # Determinar el tipo de dato
    torch_dtype = torch.float32 if dtype == "float32" else torch.float16

    # Archivo para guardar métricas de entrenamiento
    metrics_log_file = os.path.join(train_dir, "training_metrics.csv")
    with open(metrics_log_file, "w") as f:
        f.write("epoch,epoch_loss,avg_psnr,epoch_duration\n")

    # Cargar el pipeline de inpainting
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype, safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Cargar componentes: text encoder, VAE y U-Net
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
   
    # Load UNet and force float32
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device, dtype=torch.float32)
    logger.info(f"UNet forced dtype: {next(unet.parameters()).dtype}")  # Confirm float32

    # Congelar gradientes para VAE, text encoder y U-Net (se entrena solo LoRA)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(args.train_unet)
    logger.info(f"UNet training enabled: {args.train_unet}")
    if args.train_unet:
        wandb_run.tags = wandb_run.tags + ("TRAIN_UNET",)

    # Update wandb config with UNet training status
    wandb.config.update({"train_unet": args.train_unet})

    # Enviar modelos a GPU y configurar tipo de dato
    unet.to(device, dtype=torch.float32)                    #PUSE UNET EN 32
    vae.to(device, torch_dtype)
    text_encoder.to(device, torch_dtype)
    

    # Configurar LoRA en U-Net
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        init_lora_weights="gaussian",
        lora_dropout=lora_dropout,
    )

    if (overfitting):
        train_loader = overfitting_loader
        sampling_loader = overfitting_loader
        wandb_run.tags = wandb_run.tags + ("OVERFITTING",)

    wandb.config.update({"num_images": len(train_loader.dataset)})

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
    # Scheduler Cosine con warmup
    total_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10% de pasos para warmup
        num_training_steps=total_steps
    )
   
    # Noise scheduler para la difusión
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")    

    # Get first max_num_train_psnr_images samples from train_loader
    first_batch = next(iter(train_loader))
    train_set = (
        first_batch[0][:args.eval_sample_size], 
        first_batch[1][:args.eval_sample_size],
        first_batch[2][:args.eval_sample_size],
        first_batch[3][:args.eval_sample_size]
    )

    # Get first max_num_train_psnr_images samples from test_loader 
    first_batch = next(iter(test_loader))
    test_set = (
        first_batch[0][:args.eval_sample_size],
        first_batch[1][:args.eval_sample_size],
        first_batch[2][:args.eval_sample_size], 
        first_batch[3][:args.eval_sample_size]
    )

    first_time_ever = True
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Add this line to track epoch start time
        
        # Update current epoch in datasets
        train_loader.dataset.current_epoch = epoch
        val_loader.dataset.current_epoch = epoch
        test_loader.dataset.current_epoch = epoch
        rooms_with_furniture_loader.dataset.current_epoch = epoch
        sampling_loader.dataset.current_epoch = epoch
        overfitting_loader.dataset.current_epoch = epoch

        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        unet.train()
        epoch_loss = 0.0

        for input_images, input_masks, targets, unpadded_masks in tqdm(train_loader):
            batch_start_time = time.time()  # Start timing the batch

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
                evaluate_and_save_samples(pipe, train_set, test_set, rooms_with_furniture_loader, -1, train_dir)

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

            batch_end_time = time.time()  # End timing the batch
            batch_time = batch_end_time - batch_start_time
            
            logger.info(f"Batch loss: {loss.item()} | Learning rate: {lr_scheduler.get_last_lr()[0]} | Batch time: {batch_time:.2f}s")
            wandb.log({
                "batch_loss": loss.item(), 
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "batch_processing_time": batch_time,
                "epoch": epoch + 1
            })

            epoch_loss += loss.item()

        # Promedio de pérdida por época
        avg_epoch_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch + 1})
        accelerator.print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss}")

        # LR actual
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr, "epoch": epoch + 1})

        unet.eval()
        evaluate_and_save_samples(pipe, train_set, test_set, rooms_with_furniture_loader, epoch, train_dir)
        unet.train()

        wandb.log({"epoch_loss": epoch_loss / len(train_loader), "epoch": epoch + 1})
        logger.info(f"Epoch Loss: {epoch_loss / len(train_loader)}")

        # Save LoRA weights at the end of each epoch
        if save_epoch_tensors:
            accelerator.wait_for_everyone()
            unet = accelerator.unwrap_model(unet)
            epoch_weights_path = f"{train_dir}lora_weights_epoch_{epoch + 1}"
            unet.save_attn_procs(epoch_weights_path, unet_lora_layers=pipe.unet.attn_processors)
            
            # Upload safetensors to wandb if they exist
            upload_safetensors_to_wandb(f"{epoch_weights_path}/pytorch_lora_weights.safetensors", f"lora_weights_epoch_{epoch + 1}")

        # After all the epoch operations, before starting next epoch:
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Registrar métricas en CSV (ajusta PSNR si lo calculas)
        with open(metrics_log_file, "a") as f:
            f.write(f"{epoch},{avg_epoch_loss},{epoch_duration}\n")
        
        wandb.log({
            "epoch_duration": epoch_duration,
            "epoch": epoch + 1
        })
        logger.info(f"Epoch {epoch + 1} duration: {epoch_duration:.2f}s")

    # Final del entrenamiento
    accelerator.wait_for_everyone()
    accelerator.print("Training complete. Saving final LoRA weights...")
    unet = accelerator.unwrap_model(unet)
    final_weights_path = f"{train_dir}final_lora_weights"
    unet.save_attn_procs(final_weights_path, unet_lora_layers=pipe.unet.attn_processors)
    
    # Upload final safetensors to wandb
    upload_safetensors_to_wandb(f"{final_weights_path}/pytorch_lora_weights.safetensors", "final_lora_weights")


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
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="Dropout probability for LoRA layers (default: 0.1)")
    parser.add_argument("--lora-target-modules", type=str, nargs='+', default=["to_k", "to_q", "to_v", "to_out.0"], help="Target modules for LoRA adaptation")
    parser.add_argument("--guidance-scale", type=float, default=1, help="Guidance scale for inpainting")
    parser.add_argument("--inference-steps", type=int, default=20, help="Number of inference steps for inpainting")
    parser.add_argument("--overfitting", action='store_true', help="Enable overfitting mode")
    parser.add_argument("--no-overfitting", dest="overfitting", action='store_false', help="Disable overfitting mode")
    parser.set_defaults(overfitting=False)
    parser.add_argument("--save-epoch-tensors", action="store_true", help="Save LoRA weights after each epoch")
    parser.set_defaults(save_epoch_tensors=False)
    parser.add_argument("--eval-sample-size", type=int, default=5, help="Number of images to use for evaluation metrics (PSNR, SSIM, etc.)")
    parser.add_argument("--save-epoch-result-images", action="store_true", help="Save result images after each epoch")
    parser.set_defaults(save_epoch_result_images=False)
    parser.add_argument("--shape-warmup-epochs-pct", type=float, default=0.0, help="Percentage of epochs to train with basic shape masks before real masks (0.0-1.0)")
    parser.add_argument("--train-unet", action="store_true", help="Enable training of UNet")
    parser.set_defaults(train_unet=False)
    
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

    train_loader, val_loader, test_loader, overfitting_loader, sampling_loader, rooms_with_furniture_loader = load_dataset(
        inputs_dir=args.empty_rooms_dir, 
        masks_dir=args.masks_dir, 
        batch_size=args.batch_size, 
        mask_padding=0,
        img_size=args.img_size,
        num_epochs=args.epochs,
        shape_warmup_epochs_pct=args.shape_warmup_epochs_pct,
        logger=logger)

    model_id = MODELS[args.model]

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
        "inference_steps": args.inference_steps,
        "lr_scheduler": args.lr_scheduler,
        "overfitting": args.overfitting,
        "shape_warmup_epochs_pct": args.shape_warmup_epochs_pct,
        "train_unet": args.train_unet,
        })

    train_lora(model_id, 
               train_loader, 
               test_loader, 
               val_loader, 
               sampling_loader,
               overfitting_loader,
               rooms_with_furniture_loader,
               num_epochs=args.epochs,
               lr=args.initial_learning_rate,
               train_dir=train_dir,
               img_size=args.img_size, 
               save_latent_representations=args.save_latent_representations,
               save_epoch_tensors=args.save_epoch_tensors,
               lora_rank=args.lora_rank,
               lora_alpha=args.lora_alpha,
               lora_dropout=args.lora_dropout,
               lora_target_modules=args.lora_target_modules[0].split() if len(args.lora_target_modules) == 1 else args.lora_target_modules,
               dtype=args.dtype,
               overfitting=args.overfitting,
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
