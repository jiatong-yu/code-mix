#!/bin/bash

# Configuration variables
IMAGE_NAME="data-generation-image"
IMAGE_TAG="latest"
CONTAINER_NAME="data-generation-container"
HOST_PORT=8080
CONTAINER_PORT=8080

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Clean up existing container
if container_exists; then
    print_message $YELLOW "Removing existing container..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1
    docker rm $CONTAINER_NAME >/dev/null 2>&1
fi

# Build Docker image
print_message $GREEN "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} . || {
    print_message $RED "Failed to build Docker image"
    exit 1
}

# Run Docker container in interactive mode with debug options
print_message $GREEN "Starting Docker container..."

docker run -it \
    --name $CONTAINER_NAME \
    -p ${HOST_PORT}:${CONTAINER_PORT} \
    -v "$(pwd)/data:/home/jiatong/app/data" \
    -v "$(pwd)/configs:/home/jiatong/app/configs" \
    -v "$(pwd)/prompts:/home/jiatong/app/prompts" \
    -v "$(pwd)/main.py:/home/jiatong/app/main.py" \
    -v "$(pwd)/verifier_agent:/home/jiatong/app/verifier_agent" \
    -v "$(pwd)/engine:/home/jiatong/app/engine" \
    --entrypoint bash \
    ${IMAGE_NAME}:${IMAGE_TAG}

# If the container exits, show the logs
print_message $YELLOW "\nContainer exited. Showing last 50 lines of logs:"
docker logs --tail 50 $CONTAINER_NAME