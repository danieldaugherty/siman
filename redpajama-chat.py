import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'togethercomputer/RedPajama-INCITE-Chat-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if torch.cuda.is_available():
    config = {
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'load_in_8bit': True
    }
else:
    config = {'torch_dtype': torch.bfloat16}

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **config)

PROMPT_FORMAT = '''
<human>: {prompt}
<bot>:
'''

FORMAT_PROMPT_TEMPLATE = '''
<human>: Convert the following list to JSON format:
- Pull the laces tight
- Tie a knot
- Place the tips of the laces under the ball of your foot
- Pull the laces tight
- Tie a second knot
- Repeat for the other shoe
<bot>: [
  'Pull the laces tight',
  'Tie a knot',
  'Place the tips of the laces under the ball of your foot',
  'Pull the laces tight',
  'Tie a second knot',
  'Repeat for the other shoe'
]
<human>: Convert the following list to JSON format:
{prompt}
'''


def chat(prompt):
    input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = input_tokens.input_ids.shape[1]
    output = model.generate(
        input_tokens.input_ids,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id
    )
    output_str = tokenizer.decode(output.sequences[0, input_length:])
    try:
        end_index = output_str.index('<human>')
    except ValueError:
        end_index = len(output_str)
    return output_str[:end_index]


text_list = chat(PROMPT_FORMAT.format(prompt='Make a plan to aggregate real estate transactions from public data from Ulster County, NY on 2023-05-24. You have the following abilities to accomplish the goal: research, gather, format_data, and save.'))

print(text_list)

json_list = chat(FORMAT_PROMPT_TEMPLATE.format(prompt=text_list))

print(json_list)



