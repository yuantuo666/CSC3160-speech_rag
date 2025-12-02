#!/bin/bash

# ==========================================
# Remote Server Startup Script
# ==========================================

# Ensure vllm is installed
# pip install vllm

# Model name or path
# Note: Using Phi-4 model here. If using Phi-4-MM multimodal version, ensure vllm supports it or use corresponding loading method
MODEL_NAME="microsoft/Phi-4-multimodal-instruct"

# Service Port
PORT=8000

# ==========================================
# Cloudflare Tunnel Setup (Auto-forwarding)
# ==========================================

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "cloudflared not found. Installing..."
    # Assuming Linux x86_64 environment
    curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared.deb
    rm cloudflared.deb
    echo "cloudflared installed successfully."
else
    echo "cloudflared is already installed."
fi

# Start Cloudflare Tunnel in background
# Use trycloudflare free tunnel (no login required)
echo "Starting Cloudflare Tunnel for port $PORT..."
# Note: For production, consider configuring a dedicated tunnel. Using trycloudflare for demonstration.
cloudflared tunnel --url http://localhost:$PORT > tunnel.log 2>&1 &
CLOUDFLARED_PID=$!

echo "Waiting for tunnel URL..."
sleep 5
# Extract URL from log
TUNNEL_URL=$(grep -o 'https://.*\.trycloudflare.com' tunnel.log | head -n 1)

if [ -z "$TUNNEL_URL" ]; then
    echo "Warning: Could not extract tunnel URL. Please check tunnel.log."
    echo "You might need to manually check the output or ensure cloudflared is running correctly."
else
    echo "=========================================================="
    echo "Cloudflare Tunnel Started!"
    echo "Public URL: $TUNNEL_URL"
    echo "Please update REMOTE_VLLM_HOST in main.py with this URL."
    echo "Example: REMOTE_VLLM_HOST = \"$TUNNEL_URL/v1\""
    echo "=========================================================="
fi

# ==========================================
# vLLM Server Setup
# ==========================================

# Start vLLM server, compatible with OpenAI API
# --trust-remote-code: Required for some new models
# --tensor-parallel-size: Adjust based on GPU count (e.g., 1 for single card, 2 for dual cards)
# --gpu-memory-utilization: GPU memory utilization ratio, adjust as needed
echo "Starting vLLM OpenAI-compatible server for model: $MODEL_NAME on port $PORT..."

# Use trap to ensure cloudflared is closed when script exits
trap "kill $CLOUDFLARED_PID" EXIT

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --trust-remote-code \
    --port $PORT \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096
