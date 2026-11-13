import json
from pathlib import Path
import torch
from model import TransformerNMT

ROOT = Path(__file__).resolve().parent
with open(ROOT / 'checkpoints' / 'vocabularies.json', encoding='utf-8') as f:
    vocabs = json.load(f)
en_vocab, uz_vocab = vocabs['en_vocab'], vocabs['uz_vocab']
en_stoi = {token: i for i, token in enumerate(en_vocab)}
uz_stoi = {token: i for i, token in enumerate(uz_vocab)}
uz_itos = dict(enumerate(uz_vocab))
model = TransformerNMT(len(en_vocab), len(uz_vocab))
model.load_state_dict(torch.load(ROOT / 'checkpoints' / 'english_uzbek_transformer_100.pth', map_location='cpu', weights_only=True))
model.eval()

def translate(word, max_len=25):
    word = word.strip().lower()
    if not word:
        return 'Please enter an English word.'
    if any(ch not in en_stoi for ch in word):
        return 'This word contains characters outside the training vocabulary.'
    src_ids = [en_stoi['<BOS>']] + [en_stoi[ch] for ch in word] + [en_stoi['<EOS>']]
    if len(src_ids) > 100:
        return 'Word is too long.'
    src = torch.tensor([src_ids], dtype=torch.long)
    generated = [uz_stoi['<BOS>']]
    with torch.inference_mode():
        for _ in range(max_len):
            tgt = torch.tensor([generated], dtype=torch.long)
            logits = model(src, tgt)
            token_id = int(logits[0, -1].argmax().item())
            if token_id == uz_stoi['<EOS>']:
                break
            generated.append(token_id)
    result = ''.join(uz_itos[i] for i in generated[1:] if i >= 4)
    return result or '(No translation generated)'
