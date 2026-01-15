#!/bin/bash
# start.sh - Memory Optimized Startup

echo "========================================"
echo "🤖 Whisper Service - Render Free Tier"
echo "========================================"

# 🚨 CRITICAL: Memory limits for 512MB
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export TOKENIZERS_PARALLELISM=false

# Clean up /tmp
rm -rf /tmp/*.wav /tmp/models 2>/dev/null || true

echo "⚙️  Environment optimized"
echo "🚀 Starting server..."

# Start with single worker
exec uvicorn server:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level warning \
    --timeout-keep-alive 30