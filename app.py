#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import math
import os
import re

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

DEPLOY = "deploy"

# ---- load config (must match training) ----
with open(os.path.join(DEPLOY, "config.json")) as f:
    C = json.load(f)
OPTIONS = C["OPTIONS"]
PAD, UNK, CLS, SEP = C["PAD"], C["UNK"], C["CLS"], C["SEP"]
N_LEX, MAX_LEN = C["N_LEX"], C["MAX_LEN"]
D_MODEL, NHEAD, NLAYERS, DIM_FF = C["D_MODEL"], C["NHEAD"], C["NLAYERS"], C["DIM_FF"]
VOCAB_SIZE = C["vocab_size"]


# ---- exact tokenization / encoding from the notebook ----
def tokenize(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def encode_pair(prompt, option, vocab):
    p = [vocab.get(t, UNK) for t in tokenize(prompt)]
    o = [vocab.get(t, UNK) for t in tokenize(option)]
    ids = ([CLS] + p + [SEP] + o + [SEP])[:MAX_LEN]
    seg = ([0] * (len(p) + 2) + [1] * (len(o) + 1))[:MAX_LEN]
    return ids, seg


def lexical_features(prompt, option):
    pt, ot = set(tokenize(prompt)), set(tokenize(option))
    inter = len(pt & ot)
    union = len(pt | ot) or 1
    return [inter / union, inter / (len(ot) + 1), math.log1p(inter)]


# ---- architecture (identical to training) ----
class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.seg = nn.Embedding(2, D_MODEL)
        self.drop = nn.Dropout(C["DROPOUT"])
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FF,
            dropout=C["DROPOUT"], activation="gelu", batch_first=True,
            norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=NLAYERS)
        self.d_model = D_MODEL

    def forward(self, input_ids, seg_ids, pad_mask):
        S = input_ids.size(1)
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok(input_ids) + self.pos(pos) + self.seg(seg_ids))
        return self.enc(x, src_key_padding_mask=pad_mask)


class MCQModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        d = encoder.d_model
        self.norm = nn.LayerNorm(2 * d)
        self.head = nn.Sequential(
            nn.Linear(2 * d + N_LEX, d), nn.GELU(),
            nn.Dropout(C["DROPOUT"]), nn.Linear(d, 1))

    def forward(self, input_ids, seg_ids, pad_mask, lex, B, nc):
        h = self.encoder(input_ids, seg_ids, pad_mask)
        cls = h[:, 0, :]
        real = (~pad_mask).unsqueeze(-1).float()
        mean = (h * real).sum(1) / real.sum(1)
        pooled = self.norm(torch.cat([cls, mean], dim=-1))
        feat = torch.cat([pooled, lex], dim=-1)
        return self.head(feat).squeeze(-1).view(B, nc)


@st.cache_resource
def load_everything():
    with open(os.path.join(DEPLOY, "vocab.json")) as f:
        vocab = json.load(f)
    models = []
    for fold in range(1, 6):
        path = os.path.join(DEPLOY, f"fold_{fold}.pt")
        if not os.path.exists(path):
            continue
        m = MCQModel(Encoder(VOCAB_SIZE))
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        models.append(m)
    if not models:
        st.error("No fold_*.pt weights found in ./deploy — see README.")
        st.stop()
    return vocab, models


def predict(prompt, options, vocab, models):
    """Return (5,) averaged softmax probabilities over the folds."""
    f_inp, f_seg, f_lex = [], [], []
    for opt in options:
        ids, seg = encode_pair(prompt, opt, vocab)
        f_inp.append(ids); f_seg.append(seg)
        f_lex.append(lexical_features(prompt, opt))
    maxlen = max(len(x) for x in f_inp)
    N = len(f_inp)
    input_ids = np.full((N, maxlen), PAD, dtype=np.int64)
    seg_ids = np.zeros((N, maxlen), dtype=np.int64)
    pad_mask = np.ones((N, maxlen), dtype=bool)
    for i in range(N):
        L = len(f_inp[i])
        input_ids[i, :L] = f_inp[i]
        seg_ids[i, :L] = f_seg[i]
        pad_mask[i, :L] = False

    ii = torch.from_numpy(input_ids)
    sg = torch.from_numpy(seg_ids)
    pm = torch.from_numpy(pad_mask)
    lx = torch.from_numpy(np.asarray(f_lex, dtype=np.float32))

    probs = np.zeros((1, 5), dtype=np.float64)
    with torch.no_grad():
        for m in models:
            logits = m(ii, sg, pm, lx, 1, 5)      # (1,5)
            probs += F.softmax(logits, dim=1).numpy()
    return (probs / len(models))[0]


# ============================ UI ============================
st.set_page_config(page_title="MCQ Solver (from scratch)", page_icon="🧠")
st.title("🧠 MCQ Solver — from-scratch Transformer")
st.caption("A Transformer trained entirely from scratch (own vocab, MLM "
           "pretraining, 5-fold ensemble) on the Smart MCQ Solver dataset. "
           "No pretrained language models used.")

vocab, models = load_everything()
st.success(f"Loaded {len(models)}-fold ensemble | vocab size {len(vocab)}")

with st.form("mcq"):
    prompt = st.text_area("Question / prompt", height=100,
                          value="What is the powerhouse of the cell?")
    cols = st.columns(5)
    opts = []
    defaults = ["Nucleus", "Mitochondria", "Ribosome", "Golgi body",
                "Endoplasmic reticulum"]
    for i, c in enumerate(OPTIONS):
        opts.append(cols[i].text_input(f"Option {c}", value=defaults[i]))
    go = st.form_submit_button("Predict")

if go:
    probs = predict(prompt, opts, vocab, models)
    order = np.argsort(-probs)
    top3 = " ".join(OPTIONS[j] for j in order[:3])
    st.markdown(f"### Prediction (Top-3): `{top3}`")
    st.bar_chart({OPTIONS[i]: float(probs[i]) for i in range(5)})
    ranked = [(OPTIONS[j], opts[j], float(probs[j])) for j in order]
    st.table({"Rank": list(range(1, 6)),
              "Option": [r[0] for r in ranked],
              "Answer text": [r[1] for r in ranked],
              "Probability": [f"{r[2]:.3f}" for r in ranked]})
    st.caption("Top-3 string is the Kaggle submission format for this row.")