# 🧠 Smart MCQ Ranker

A multiple-choice question solver powered by a Transformer we built and trained
, no pretrained language models, no external data.
This repo hosts the live demo: type a question and five options, and the model
ranks them.

---

## What this model is

Given a question and five options (A–E), the model scores each option and
returns the three most likely answers in ranked order (MAP@3 format).

What makes it different is every weight is learned from the data
alone. There is no BERT, no GPT, no downloaded embeddings.

- **Own vocabulary** built from the dataset's text (word-level tokenizer).
- **Custom Transformer encoder** — token + positional + segment embeddings,
  4 layers, 4 attention heads, trained from random initialization.
- **Self-supervised pretraining (MLM):** the encoder first learns language
  structure by predicting masked words in the dataset's own text (BERT-style),
  before ever seeing the answer labels.
- **Cross-encoder scoring head:** each `[CLS] question [SEP] option [SEP]` pair
  is pooled (CLS + mean) and combined with a few lexical-overlap features to
  produce one score per option; a softmax over the five picks the answer.
- **5-fold ensemble + EMA:** five models are trained on different data splits,
  their weights smoothed with an exponential moving average, and their
  predictions averaged for the final answer.

## How the demo works

The app loads the saved vocabulary and all five trained fold-models, rebuilds
the exact training architecture, and **averages the five models' probabilities**
It runs on CPU; the model is small and fast.

## Repository layout

```
.
├── app.py             # Streamlit web app (inference + UI)
├── requirements.txt   # dependencies
├── scratch.ipynb      # model training 
└── deploy/
    ├── vocab.json     # vocabulary built from the dataset
    ├── config.json    # model hyperparameters
    └── fold_1.pt … fold_5.pt   # trained weights for each fold

```

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## A note on what to expect

This is a scratch model trained on a small dataset, so it answers well on
familiar topics and struggles with genuinely novel questions — it has no
outside world knowledge to draw on. That trade-off is the point: the project
demonstrates the full pipeline (tokenizer → pretraining → fine-tuning →
ensembling) built end-to-end, not a state-of-the-art accuracy score.

## Built with

Python · PyTorch · scikit-learn · Streamlit · Weights & Biases (experiment
tracking during training)