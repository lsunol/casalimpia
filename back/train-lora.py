import torch
from diffusers import StableDiffusionInpaintPipeline
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTokenizer
import os
from PIL import Image
from datetime import datetime


# Configuración básica
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

MODEL_NAME = "stabilityai/stable-diffusion-2-inpainting"
LORA_RANK = 4  # Rango del LoRA (ajusta según necesidad)
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4

# Cargar el modelo base de Stable Diffusion Inpainting
pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_NAME).to(DEVICE)
unet = pipe.unet
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

# Añadir LoRA al UNet (Low-Rank Adaptation)
def add_lora(unet, rank=LORA_RANK):
    for name, module in unet.named_modules():
        if "attn2" in name:  # Aplicar LoRA a las capas de atención
            original_weight = module.to_q.weight
            lora_down = torch.nn.Linear(original_weight.shape[1], rank, bias=False).to(DEVICE)
            lora_up = torch.nn.Linear(rank, original_weight.shape[0], bias=False).to(DEVICE)
            module.to_q.weight = torch.nn.Parameter(original_weight + lora_up(lora_down.weight))
    return unet

unet = add_lora(unet)

# Dataset personalizado
class InpaintingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.mask_files = os.listdir(mask_dir)

    def __len__(self):
        return min(len(self.image_files), len(self.mask_files))

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx % len(self.mask_files)])  # Reutilizar máscaras

        image = transforms.ToTensor()(Image.open(image_path).convert("RGB"))
        mask = transforms.ToTensor()(Image.open(mask_path).convert("L"))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Transformaciones (ajusta según tus necesidades)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Cargar datos
dataset = InpaintingDataset(
    image_dir="./data/emptyRooms",
    mask_dir="./data/automaticMasks",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizador
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

# Entrenamiento
for epoch in range(EPOCHS):
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Generar texto de prompt (puedes personalizarlo)
        prompts = ["a photo of an empty room"] * BATCH_SIZE
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        text_embeddings = text_encoder(inputs.input_ids).last_hidden_state

        # Forward pass
        with torch.no_grad():
            latents = pipe.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215  # Escalar latentes

        # Aplicar máscara
        masked_latents = latents * (1 - masks)

        # Predicción
        noise_pred = unet(masked_latents, torch.zeros_like(masked_latents), text_embeddings).sample

        # Calcular pérdida (MSE entre la predicción y los latentes originales)
        loss = torch.nn.functional.mse_loss(noise_pred, latents)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}")

# Obtener la fecha y hora actual
current_time = datetime.now().strftime("%Y.%m.%d-%H.%M")

# Nombre del archivo con fecha y hora
filename = f"{current_time}-lora-weights.safetensors"

# Guardar los pesos del LoRA
lora_weights = {name: param for name, param in unet.named_parameters() if "lora" in name}
save_file(lora_weights, filename)
print(f"LoRA weights saved to {filename}")