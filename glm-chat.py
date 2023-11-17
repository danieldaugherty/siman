from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).to('cpu', dtype=float)


def chat(prompt, history=None):
    history = history or []
    response, history = model.chat(tokenizer, prompt, history=history)
    return history.append(response)


print(chat('What is the best ice cream flavor?'))
