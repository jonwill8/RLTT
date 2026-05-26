#!/bin/bash

# Resolve the Ouro thinking model for baseline evaluation runners.
# Set MODEL_SIZE=1.4B or MODEL_SIZE=2.6B, or pass MODEL_PATH directly.
if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_SIZE=${MODEL_SIZE:-"2.6B"}

    case "$MODEL_SIZE" in
        "1.4B"|"1.4b"|"1.4")
            MODEL_SIZE="1.4B"
            MODEL_PATH="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-1.4B-Thinking"
            ;;
        "2.6B"|"2.6b"|"2.6")
            MODEL_SIZE="2.6B"
            MODEL_PATH="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking"
            ;;
        *)
            echo "ERROR: Unknown MODEL_SIZE: $MODEL_SIZE"
            echo "Valid options: 1.4B, 2.6B (or set MODEL_PATH directly)"
            exit 1
            ;;
    esac
else
    if [ -z "${MODEL_SIZE:-}" ]; then
        case "$MODEL_PATH" in
            *Ouro-1.4B-Thinking*) MODEL_SIZE="1.4B" ;;
            *Ouro-2.6B-Thinking*) MODEL_SIZE="2.6B" ;;
            *) MODEL_SIZE="custom" ;;
        esac
    else
        case "$MODEL_SIZE" in
            "1.4B"|"1.4b"|"1.4") MODEL_SIZE="1.4B" ;;
            "2.6B"|"2.6b"|"2.6") MODEL_SIZE="2.6B" ;;
        esac
    fi
fi

if [ "$MODEL_SIZE" = "custom" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH")
else
    MODEL_NAME="Ouro-${MODEL_SIZE}-Thinking"
fi
MODEL_TAG="ouro_${MODEL_SIZE}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

export MODEL_SIZE
export MODEL_PATH
export MODEL_NAME
export MODEL_TAG
