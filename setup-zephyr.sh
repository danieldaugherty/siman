#!/bin/zsh
pip install -U huggingface_hub
huggingface-cli download TheBloke/zephyr-7B-beta-GGUF zephyr-7b-beta.Q5_K_M.gguf
pip uninstall ctransformers --yes
CT_METAL=1
pip install ctransformers --no-binary ctransformers



