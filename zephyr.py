from ctransformers import AutoModelForCausalLM
from time import time

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/zephyr-7B-beta-GGUF",
    model_file="zephyr-7b-beta.Q5_K_M.gguf",
    model_type="mistral",
    gpu_layers=50,
    local_files_only=True,
    context_length=1024 * 8,
    stream=True
)


def chat(chat_history):
    start_time = time()
    result = ''
    for token in llm(chat_history):
        result += token
        print(token, end='')
    print(f"\nTime taken: {time() - start_time}")
    return result

