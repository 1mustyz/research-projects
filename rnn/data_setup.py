import torch

# Small corpus — keep it short so training is fast to observe
text = """
the quick brown fox jumps over the lazy dog.
she sells seashells by the seashore.
how much wood would a woodchuck chuck if a woodchuck could chuck wood.
peter piper picked a peck of pickled peppers.
the rain in spain stays mainly on the plain.
""".strip().lower()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Corpus length: {len(text)} characters")
print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Vocabulary: {chars}")

def encode(s):
    return torch.tensor([char_to_idx[c] for c in s], dtype=torch.long)

def decode(indices):
    return ''.join(idx_to_char[i] for i in indices)

data = encode(text)
print(f"\nEncoded data shape: {data.shape}")
print(f"First 20 chars encoded: {data[:20].tolist()}")
print(f"Decoded back: '{decode(data[:20].tolist())}'")