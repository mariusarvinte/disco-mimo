#!/bin/bash

podman run -it \
  --device nvidia.com/gpu=all \
  -v ~/disco-mimo/podman/pretty.bashrc:/home/user/.bashrc \
  disco-mimo:latest bash
