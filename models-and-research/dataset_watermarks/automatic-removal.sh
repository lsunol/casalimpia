#!/bin/bash
# This script will use unwatermark.ai api to remove watermark from images
# Requirements:
# curl, jq

LOG_FILE="log_$(date +%Y%m%d_%H%M%S).txt"
LOG_DEBUG_FILE="unwatermark-script.log"

# Set default values
debug_enabled=false
TIMEOUT=30
FETCH_INTERVAL=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            debug_enabled=true
            shift
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --fetch-interval)
            FETCH_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [-d|--debug] [-i|--input dir] [-o|--output dir] [--timeout seconds] [--fetch-interval seconds]"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Input and output directories are required"
    echo "Usage: $0 [-d|--debug] [-i|--input dir] [-o|--output dir] [--timeout seconds] [--fetch-interval seconds]"
    exit 1
fi

# Crear el directorio de salida si no existe
mkdir -p "$OUTPUT_DIR"

# Crear archivo de log
echo "Log de ejecución iniciado el: $(date)" > "$LOG_FILE"

# Función para mostrar el progreso
log_progress() {
    local message="$1"
    local status="$2"
    
    # Add status suffix if provided
    if [ -n "$status" ]; then
        message="$message [${status}]"
    fi
    
    # Add newline unless status is empty
    if [ -z "$status" ]; then
        echo -n "$message" | tee -a "$LOG_FILE"
    else
        echo "$message" | tee -a "$LOG_FILE"
    fi
}

# Function to handle debug messages
debug_log() {

    MESSAGE="[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: $1"

    if [ "$debug_enabled" = true ]; then
        echo $MESSAGE
    fi
    echo $MESSAGE >> "$LOG_DEBUG_FILE"
}

# Initialize first product serial
PRODUCT_SERIAL=$(openssl rand -hex 16)
debug_log "Generated first product serial: $PRODUCT_SERIAL"

# Print recap of options
echo "=== Configuration ==="
echo "Debug mode: ${debug_enabled} (debug file: $PWD/$LOG_DEBUG_FILE)"
echo "Timeout: ${TIMEOUT} seconds"
echo "Fetch interval: ${FETCH_INTERVAL} seconds"
echo "Product serial: $PRODUCT_SERIAL"

# Count files in directories
input_count=$(find "$INPUT_DIR" -type f -name "*.jpg" | wc -l)
output_count=$(find "$OUTPUT_DIR" -type f -name "*.jpg" | wc -l)

echo "Input directory: ${INPUT_DIR} (${input_count} images)"
echo "Output directory: ${OUTPUT_DIR} (${output_count} images)"
echo "==================="
echo

# Wait for user confirmation before starting
echo "Press ENTER to start fetching..."
read -r

# Contar el número total de imágenes
TOTAL_IMAGES=$(find "$INPUT_DIR" -type f -name "*.jpg" | wc -l)
PROCESSED_COUNT=0

for IMAGE_PATH in "$INPUT_DIR"/*.jpg; do

  PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
  IMAGE_NAME=$(basename "$IMAGE_PATH")
  debug_log "Processing image $PROCESSED_COUNT: $IMAGE_NAME"

  # Add this after IMAGE_NAME=$(basename "$IMAGE_PATH")
  OUTPUT_IMAGE_PATH="$OUTPUT_DIR/$IMAGE_NAME"
  debug_log "OUTPUT IMAGE PATH: $OUTPUT_IMAGE_PATH"

  if [ -f "$OUTPUT_IMAGE_PATH" ]; then
    log_progress "($(printf '%3d' $PROCESSED_COUNT)/$(printf '%3d' $TOTAL_IMAGES) - $(printf '%3d' $(( PROCESSED_COUNT * 100 / TOTAL_IMAGES )))%) $(printf '%-50s' "$IMAGE_NAME: ")" "ALREADY PROCESSED"
    continue
  fi

  # generate product serial
  if [ $((PROCESSED_COUNT % 50)) -eq 0 ]; then
    PRODUCT_SERIAL=$(openssl rand -hex 16)
    debug_log "Generated new product serial for next 50 iterations: $PRODUCT_SERIAL"
  fi

  debug_log "Using product serial: $PRODUCT_SERIAL"

  # Realizar la llamada para crear el job
  CREATE_JOB_RESPONSE=$(curl --silent --location 'https://api.unwatermark.ai/api/unwatermark/v4/ai-remove-auto/create-job' \
    --header 'accept: */*' \
    --header 'accept-language: es-ES,es;q=0.9,ca;q=0.8,en;q=0.7' \
    --header 'authorization;' \
    --header 'origin: https://unwatermark.ai' \
    --header 'priority: u=1, i' \
    --header 'product-code: 067003' \
    --header 'product-serial: '"$PRODUCT_SERIAL$PRODUCT_SERIAL_SUFFIX" \
    --header 'referer: https://unwatermark.ai/' \
    --header 'sec-ch-ua: "Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"' \
    --header 'sec-ch-ua-mobile: ?0' \
    --header 'sec-ch-ua-platform: "Linux"' \
    --header 'sec-fetch-dest: empty' \
    --header 'sec-fetch-mode: cors' \
    --header 'sec-fetch-site: same-site' \
    --header 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36' \
    --form "original_image_file=@\"$IMAGE_PATH\"")

  debug_log "create-job result: $CREATE_JOB_RESPONSE"

  # Check if code is 300002 (out of times)
  RESPONSE_CODE=$(echo "$CREATE_JOB_RESPONSE" | jq -r '.code')
  if [ "$RESPONSE_CODE" -eq 300002 ]; then
    if command -v nordvpn >/dev/null 2>&1 && [ -f nordvpn-countries.txt ]; then
      nordvpn disconnect
      nordvpn connect
      sleep 15  # Wait for connection to establish
    fi
    continue
  fi
  
  JOB_ID=$(echo "$CREATE_JOB_RESPONSE" | jq -r '.result.job_id')
  debug_log "Job ID: $JOB_ID"

  if [ "$JOB_ID" == "null" ]; then
    log_progress "" "ERROR (no job ID)"

    continue
  fi

  log_progress "($(printf '%3d' $PROCESSED_COUNT)/$(printf '%3d' $TOTAL_IMAGES) - $(printf '%3d' $(( PROCESSED_COUNT * 100 / TOTAL_IMAGES )))%) $(printf '%-50s' "$IMAGE_NAME: Processing")" 

  # Esperar a que el proceso termine
  TIME_WAITED=0
  while :; do
    sleep $FETCH_INTERVAL
    TIME_WAITED=$((TIME_WAITED + $FETCH_INTERVAL))

    STATUS_RESPONSE=$(curl --silent --location "https://api.unwatermark.ai/api/unwatermark/v4/ai-remove-auto/get-job/$JOB_ID" \
      --header 'accept: */*' \
      --header 'accept-language: es-ES,es;q=0.9,ca;q=0.8,en;q=0.7' \
      --header 'authorization;' \
      --header 'origin: https://unwatermark.ai' \
      --header 'priority: u=1, i' \
      --header 'product-code: 067003' \
      --header 'referer: https://unwatermark.ai/' \
      --header 'sec-ch-ua: "Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"' \
      --header 'sec-ch-ua-mobile: ?0' \
      --header 'sec-ch-ua-platform: "Windows"' \
      --header 'sec-fetch-dest: empty' \
      --header 'sec-fetch-mode: cors' \
      --header 'sec-fetch-site: same-site' \
      --header 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

    debug_log "get-job result: $STATUS_RESPONSE"

    STATUS_CODE=$(echo "$STATUS_RESPONSE" | jq -r '.code')
    
    # Check if STATUS_CODE is a valid number
    if [[ "$STATUS_CODE" =~ ^[0-9]+$ ]] && [ "$STATUS_CODE" -eq 100000 ]; then
      output_image_url=$(echo "$STATUS_RESPONSE" | jq -r '.result.output_image_url[0]')
      debug_log "Output image URL: $output_image_url"
      break
    fi

    if [ "$TIME_WAITED" -ge $TIMEOUT ]; then
      log_progress "" "TIMEOUT"
      continue 2
    fi

  done

  # Descargar la imagen procesada
  OUTPUT_IMAGE_PATH="$OUTPUT_DIR/$IMAGE_NAME"
  curl --silent --location "$output_image_url" \
    --header 'accept: image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8' \
    --header 'accept-language: es-ES,es;q=0.9,ca;q=0.8,en;q=0.7' \
    --header 'priority: u=1, i' \
    --header 'referer: https://unwatermark.ai/' \
    --header 'sec-ch-ua: "Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"' \
    --header 'sec-ch-ua-mobile: ?0' \
    --header 'sec-ch-ua-platform: "Windows"' \
    --header 'sec-fetch-dest: image' \
    --header 'sec-fetch-mode: no-cors' \
    --header 'sec-fetch-site: cross-site' \
    --header 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36' \
    --output "$OUTPUT_IMAGE_PATH"

  image_size=$(stat --printf="%s" "$OUTPUT_IMAGE_PATH" | awk '{printf "%.2f", $1/1024}')
  debug_log "Downloaded image size: $image_size kb"
  log_progress "($image_size kb)" "OK"

done

log_progress "Proceso terminado. Imágenes procesadas: $PROCESSED_COUNT/$TOTAL_IMAGES."
