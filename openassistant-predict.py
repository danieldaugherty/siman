import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, max_memory='40GB', torch_dtype=torch.bfloat16)


def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=10.0, temperature=0.1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('result', result)
    return result

predict('''
The following is a search engine query for finding where public real estate records are available for download for Ulster County, NY:

''')
