"""
Prepare the SS-Safari chatbot dataset for character-level language modeling.
Instead of GPT-2 BPE, we map characters to ints.
This version fetches the dataset from a URL.
"""

import os
import pickle
import numpy as np
import requests

# === CUSTOM INPUT ===
dataset_url = 'https://raw.githubusercontent.com/your-username/your-repo/main/ss_safari_chat.txt'  # 🔁 Replace with your actual raw GitHub file link

# === DOWNLOAD DATA ===
response = requests.get(dataset_url)
if response.status_code != 200:
    raise Exception(f"Failed to download dataset. Status code: {response.status_code}")
data = response.text
print(f"✅ Dataset downloaded successfully. Length in characters: {len(data):,}")

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

# === SAVE OUTPUT FILES ===
output_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
train_path = os.path.join(output_dir, 'train.bin')
val_path = os.path.join(output_dir, 'val.bin')
meta_path = os.path.join(output_dir, 'meta.pkl')

train_ids.tofile(train_path)
val_ids.tofile(val_path)

# Save vocab and mappings
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"✅ Tokenization complete. Files saved: {train_path}, {val_path}, {meta_path}")
