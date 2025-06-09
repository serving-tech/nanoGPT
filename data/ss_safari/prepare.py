"""
Prepare the SS-Safari chatbot dataset for character-level language modeling.
Instead of GPT-2 BPE, we map characters to ints.
"""

import os
import pickle
import numpy as np

# === CUSTOM INPUT ===
input_file_path = os.path.join(os.path.dirname(__file__), 'ss_safari_chat.txt')
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"Dataset file not found at: {input_file_path}")

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"Length of dataset in characters: {len(data):,}")

# Unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("Unique characters:", ''.join(chars))
print(f"Vocab size: {vocab_size}")

# Char-to-ID and ID-to-Char mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Split the data into training and validation
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.1):]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Save bin files
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save vocab and mappings
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("✅ Tokenization complete. Files saved: train.bin, val.bin, meta.pkl")
