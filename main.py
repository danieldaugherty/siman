from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, repeat_penalty=10)
    print('outputs', outputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('result', result)
    return result

predict('''
What's the best ice cream flavor?
''')
