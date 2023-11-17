import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = 'openlm-research/open_llama_3b_600bt_preview'
# model_path = 'openlm-research/open_llama_7b_700bt_preview'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map='auto',
)

prompt = '''
Q: Generate a search engine query for finding where public real estate records are available for download for Ulster County, NY.
A: 
'''

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=64,
    temperature=0.1
)
print(tokenizer.decode(generation_output[0]))
