from ctransformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

config = AutoConfig.from_pretrained(
    'TheBloke/MPT-7B-Instruct-GGML',
    stream=True,
    max_new_tokens=256,
    repetition_penalty=10.0,
    temperature=0.2
)
model = AutoModelForCausalLM.from_pretrained(
    './models/mpt-7b-instruct.ggmlv3.q5_1.bin',
    model_type='mpt',
    config=config
)

# PROMPT_TEMPLATE = """\
# Below is an instruction that describes a task. Write a response that completes the request appropriately.
# ### Instruction:
# {prompt}
# ### Response:
# """

PLAN_PROMPT_TEMPLATE = '''\
Below is an instruction that describes a task. Write a response that completes the request appropriately.
### Instruction:
PlannerBot
  plan() // generate list of steps to accomplish a goal; output in CSV format

ResearcherBot
  research() // discover information needed to carry out more actions

AggregatorBot
  gather() // obtain desired data to aggregate
  save() // save aggregated data to DB
  
Constraints
  PlannerBot.plan responses must be in CSV format
  PlannerBot.plan responses must only include the list of steps
  PlannerBot.plan steps have two attributes: assignee, description
  PlannerBot.plan step assignees must be a bot declared above
  
Example

PlannerBot.plan({prompt}) // returns a Plan

### Response:
'''

FORMAT_PROMPT_TEMPLATE='''
Below you'll take a plan and convert it to a JSON representation of each step.
### Instruction:
{prompt}
### Response:
'''


def instruct(prompt):
    input = model.tokenize(prompt)
    output = model.generate(input)
    output_text = ''
    for token in tqdm(output):
        decoded_token = model.detokenize(token)
        output_text += decoded_token
    return output_text

plan = instruct(PLAN_PROMPT_TEMPLATE.format(prompt='Aggregate all real estate transaction data for Ulster County, NY yesterday.'))

print(plan)

json = instruct(FORMAT_PROMPT_TEMPLATE.format(prompt=plan))

print(json)
