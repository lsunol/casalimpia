import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import cv2

# MASK POST-PROCESSING
def clean_mask(mask, kernel_size=9):                            # Mejorar filtrado del clean => remove noise
    kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)
    cleaned = scipy.ndimage.binary_opening(mask, structure=kernel).astype(numpy.uint8)      #   Eliminates small isolated region
    cleaned = scipy.ndimage.binary_closing(cleaned, structure=kernel).astype(numpy.uint8)   #   Fill small gaps in the mask
    return cleaned                                      # Limpieza más agresiva para eliminar ruido

def smooth_mask(mask, blur_size=9):  # Suavizado para eliminar ruido
    smoothed = scipy.ndimage.gaussian_filter(mask.astype(numpy.float32), sigma=blur_size)
    return (smoothed > 0.3).astype(numpy.uint8)  # Ajustar el umbral para suavizado

def close_mask(mask, kernel_size=9):  # Mejorar cierre para huecos internos
    kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)
    closed = scipy.ndimage.binary_closing(mask, structure=kernel).astype(numpy.uint8)
    closed = scipy.ndimage.binary_dilation(closed, structure=kernel).astype(numpy.uint8)  # Expandir bordes ligeramente
    return closed

def refine_contours(mask):
    contours, _ = cv2.findContours(mask.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_mask = numpy.zeros_like(mask, dtype=numpy.uint8)
    for contour in contours:
        hull = cv2.convexHull(contour, returnPoints=True)
        cv2.drawContours(refined_mask, [hull], -1, 255, thickness=cv2.FILLED)
    return refined_mask / 255  # Refinar sin aproximaciones exageradas

def get_background_classes():
    return {
        'wall': 1,
        'floor': 4,
        'ceiling': 6,
        'windowpane': 9,
        'door': 15
    }

def create_foreground_mask(pred, pred_classes, names, background_classes):
    foreground_mask = numpy.ones_like(pred, dtype=numpy.uint8)
    print("\nDetected objects:")
    for c in pred_classes[:15]:
        pixel_count = numpy.sum(pred == c)
        percentage = (pixel_count / pred.size) * 100
        class_name = names[c+1] if c+1 in names else f'Class {c}'
        is_background = (c + 1) in background_classes.values()
        type_label = "BACKGROUND" if is_background else "FOREGROUND"
        print(f"{class_name}: {percentage:.2f}% of image ({type_label})")
        if is_background:
            foreground_mask[pred == c] = 0
    return foreground_mask

def process_image(image_path, output_dir):
    print("Initializing model...")
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated', fc_dim=2048, weights=r'C:\Users\alexs\Desktop\UPC_SCHOOL\IA\PROJECT\SAM\ADE20K+SAM\model_checkpoints\encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup', fc_dim=2048, num_class=150, weights=r'C:\Users\alexs\Desktop\UPC_SCHOOL\IA\PROJECT\SAM\ADE20K+SAM\model_checkpoints\decoder_epoch_20.pth', use_softmax=True)
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).to('cuda')
    segmentation_module.eval()

    print("Processing image...")
    pil_image = PIL.Image.open(image_path).convert('RGB')
    img_original = numpy.array(pil_image)
    img_data = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(pil_image).to('cuda')
    singleton_batch = {'img_data': img_data[None]}
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=img_data.shape[1:])
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    names = {}
    with open(r'C:\Users\alexs\Desktop\UPC_SCHOOL\IA\PROJECT\SAM\ADE20K+SAM\mask_colorcoding\object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    background_classes = get_background_classes()
    foreground_mask = create_foreground_mask(pred, numpy.bincount(pred.flatten()).argsort()[::-1], names, background_classes)

    # Generar las diferentes máscaras a partir del ROW
    raw_mask = (foreground_mask * 255).astype(numpy.uint8)
    cleaned_mask = clean_mask(foreground_mask, kernel_size=7)
    smoothed_mask = smooth_mask(foreground_mask, blur_size=7)
    closed_mask = close_mask(foreground_mask, kernel_size=9)
    refined_mask = refine_contours(foreground_mask)

    # Combinar cleaned + closed + smoothed
    combined_mask = numpy.maximum(cleaned_mask, numpy.maximum(closed_mask, smoothed_mask))

    # Combinar RAW + refined utilizando un enfoque lógico
    raw_refined_combined = numpy.logical_or(raw_mask > 0, refined_mask > 0).astype(numpy.uint8) * 255

    # Nueva máscara: RAW -> CLOSED -> CLEANED -> SMOOTHED
    raw_to_smoothed = smooth_mask(clean_mask(close_mask(foreground_mask, kernel_size=9), kernel_size=7), blur_size=7)

    # Imprimir estadísticas para verificar diferencias
    print("\nMask Statistics:")
    print(f"Raw mask active pixels: {numpy.sum(raw_mask > 0)}")
    print(f"Cleaned mask active pixels: {numpy.sum(cleaned_mask > 0)}")
    print(f"Smoothed mask active pixels: {numpy.sum(smoothed_mask > 0)}")
    print(f"Closed mask active pixels: {numpy.sum(closed_mask > 0)}")
    print(f"Refined mask active pixels: {numpy.sum(refined_mask > 0)}")
    print(f"Combined mask active pixels: {numpy.sum(combined_mask > 0)}")
    print(f"Raw + Refined combined mask active pixels: {numpy.sum(raw_refined_combined > 0)}")
    print(f"Raw -> Closed -> Cleaned -> Smoothed active pixels: {numpy.sum(raw_to_smoothed > 0)}")

    # Guardar resultados
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    PIL.Image.fromarray(img_original).save(os.path.join(output_dir, f"{base_filename}_original.png"))
    PIL.Image.fromarray(raw_mask).save(os.path.join(output_dir, f"{base_filename}_mask_raw.png"))
    PIL.Image.fromarray((cleaned_mask * 255).astype(numpy.uint8)).save(os.path.join(output_dir, f"{base_filename}_mask_cleaned.png"))
    PIL.Image.fromarray((smoothed_mask * 255).astype(numpy.uint8)).save(os.path.join(output_dir, f"{base_filename}_mask_smoothed.png"))
    PIL.Image.fromarray((closed_mask * 255).astype(numpy.uint8)).save(os.path.join(output_dir, f"{base_filename}_mask_closed.png"))
    PIL.Image.fromarray((refined_mask * 255).astype(numpy.uint8)).save(os.path.join(output_dir, f"{base_filename}_mask_refined.png"))
    PIL.Image.fromarray((combined_mask * 255).astype(numpy.uint8)).save(os.path.join(output_dir, f"{base_filename}_mask_combined.png"))
    PIL.Image.fromarray(raw_refined_combined).save(os.path.join(output_dir, f"{base_filename}_mask_raw_refined_combined.png"))
    PIL.Image.fromarray((raw_to_smoothed * 255).astype(numpy.uint8)).save(os.path.join(output_dir, f"{base_filename}_mask_raw_to_smoothed.png"))
    print(f"Masks saved for {base_filename}.")

if __name__ == "__main__":
    input_dir = r"C:\Users\alexs\Desktop\UPC_SCHOOL\IA\PROJECT\SAM\ADE20K+SAM\images"
    output_dir = r"C:\Users\alexs\Desktop\UPC_SCHOOL\IA\PROJECT\SAM\ADE20K+SAM\output"
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        process_image(image_path, output_dir)

