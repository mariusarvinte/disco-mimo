#!/bin/bash

cat agent.py | \
  podman run -i --rm \
    --device nvidia.com/gpu=all \
    -e OPENROUTER_API_KEY \
    disco-mimo:latest uv run --group torch --group agent -