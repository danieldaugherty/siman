from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
# model = SentenceTransformer('all-mpnet-base-v2')

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed(sentence):
    # Tokenize sentences
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings_normalized = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings_normalized


def compare_sentences():
    # Sentences we want sentence embeddings for
    sentences = [
        'I have a truck.',
        'I like to drive.',
        'Turn left at the stop sign.',
        'Turn off your headlights.',
        'A driving wind battered the children.',
        'Motivation is a vehicle for action.',
        'He used sign language.',
        'Star light, star bright—let me make a wish tonight.',
        'Yeah, chocolate!',
        'Bunny rabbits like to bounce.',
        'He was an odd sorta fella.'
    ]
    reference_sentence = sentences[0]
    ref_embedding = embed(reference_sentence)
    for sentence in sentences:
        sentence_embeddings = embed(sentence)
        cosine_scores = util.cos_sim(sentence_embeddings, ref_embedding)
        print(sentence, cosine_scores)


compare_sentences()
