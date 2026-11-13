---
title: LingoUz
emoji: 🌈
colorFrom: pink
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# 🌈 LingoUz — English → Uzbek Transformer

A colorful web demo of a **character-level encoder–decoder Transformer** trained from scratch on 100 English–Uzbek word pairs. Based on the main architecture of *Attention Is All You Need* (Vaswani et al., 2017), with a smaller model (2 layers, 4 heads, `d_model=64`).

## Features
- Real PyTorch Transformer inference (no hardcoded dictionary fallback)
- Autoregressive greedy character decoding
- Responsive Gradio interface
- 100-word educational training dataset

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Before running, put your **own trained** `english_uzbek_transformer_100.pth` and `vocabularies.json` in `checkpoints/`.

## Model & limitations

The dataset has only 100 training pairs. Training-set accuracy reflects memorization, **not** generalization to unseen words or sentences. This is not a production translator. No external pretrained model is used.

## Dataset and evaluation

`data/english_uzbek_100.csv` contains the 100 paired words. Add measured training word accuracy after running the evaluation cell. The model checkpoint must use the exact character vocabularies from training.

## Reference

Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
