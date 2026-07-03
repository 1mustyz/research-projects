# Handwritten RNN — Character-Level Language Model from Scratch

A minimal PyTorch implementation of a Vanilla RNN that trains a character-level language model — written **without** `nn.RNN` or any other recurrent black box. Every recurrence step is coded explicitly so the mechanics are fully transparent.

---

## Project Structure

```
rnn/
├── data_setup.py          # Corpus, vocabulary, encoding/decoding utilities
├── data_setup.ipynb       # Same data setup, explored interactively in notebook form
├── handwritten_rnn.ipynb  # Vanilla RNN: model, training loop, generation (annotated notebook)
├── handwritten_rnn.pt     # Saved Vanilla RNN weights (produced after training)
├── lstm_version.ipynb     # LSTM version of the same model, using nn.LSTM (annotated notebook)
└── README.md              # This file
```

---

## Background

An RNN processes a sequence one element at a time. At each step $t$ it combines the current input $x_t$ with its own previous hidden state $h_{t-1}$ to produce a new hidden state:

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

The hidden state is then projected to output logits:

$$y_t = W_{hy} \cdot h_t + b_y$$

The hidden state $h_t$ is the model's **memory** — it carries a compressed summary of all characters seen so far. This project makes both equations directly visible in Python code rather than hiding them inside a library call.

---

## File Reference

### `data_setup.py`

Prepares the text corpus for training.

| Symbol            | Type               | Description                                                                     |
| ----------------- | ------------------ | ------------------------------------------------------------------------------- |
| `text`            | `str`              | Raw lowercased corpus (~270 characters of English tongue-twisters and pangrams) |
| `chars`           | `list[str]`        | Sorted list of every unique character that appears in `text`                    |
| `vocab_size`      | `int`              | `len(chars)` — total number of unique characters                                |
| `char_to_idx`     | `dict[str, int]`   | Maps each character to its integer index                                        |
| `idx_to_char`     | `dict[int, str]`   | Maps each integer index back to its character                                   |
| `encode(s)`       | `function`         | Converts a string to a `torch.LongTensor` of character indices                  |
| `decode(indices)` | `function`         | Converts a list/tensor of indices back to a string                              |
| `data`            | `torch.LongTensor` | The entire corpus pre-encoded as integer indices                                |

The corpus used:

```
the quick brown fox jumps over the lazy dog.
she sells seashells by the seashore.
how much wood would a woodchuck chuck if a woodchuck could chuck wood.
peter piper picked a peck of pickled peppers.
the rain in spain stays mainly on the plain.
```

---

### `handwritten_rnn.ipynb` — Cell-by-Cell Explanation

#### Cell 1 — Complete Reference Script

The full implementation as a single runnable Python script. All subsequent cells reproduce this code in structured, annotated sections.

---

#### Cell 2 — Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_setup import text, chars, vocab_size, char_to_idx, idx_to_char, encode, decode, data
```

| Import                | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `torch`               | Core tensor operations, autograd, RNG                          |
| `torch.nn`            | Parameterised layers: `nn.Linear`, `nn.Embedding`, `nn.Module` |
| `torch.nn.functional` | Stateless functions: `F.softmax`, `F.cross_entropy`            |
| `data_setup.*`        | Corpus data and vocabulary (see above)                         |

---

#### Cell 3 — Random Seed

```python
torch.manual_seed(42)
```

Fixes PyTorch's random number generator so results are reproducible across runs:

- `nn.Linear` and `nn.Embedding` weight initialisation draws from the RNG
- `torch.randint` in `get_batch` uses the RNG for sampling start positions

---

#### Cell 4 — `VanillaRNN` Model

```python
class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size): ...
    def forward(self, x, h=None): ...
```

**Constructor — `__init__`**

| Layer   | Type                  | I/O Shape                   | Role                                                  |
| ------- | --------------------- | --------------------------- | ----------------------------------------------------- |
| `embed` | `nn.Embedding`        | `vocab_size → hidden_size`  | Turns a character index into a trainable dense vector |
| `Wxh`   | `nn.Linear`           | `hidden_size → hidden_size` | Input-to-hidden weights (includes bias $b_h$)         |
| `Whh`   | `nn.Linear` (no bias) | `hidden_size → hidden_size` | Hidden-to-hidden weights — the recurrent connection   |
| `Why`   | `nn.Linear`           | `hidden_size → vocab_size`  | Hidden-to-output projection                           |

**Forward pass — `forward(x, h)`**

```
x: (batch, seq_len)  ← integer character indices

1. embed(x)  →  (batch, seq_len, hidden_size)
2. For t in 0 … seq_len-1:
       x_t = embedded[:, t, :]                         # current char vector
       h   = tanh( Wxh(x_t) + Whh(h) )                # ← explicit recurrence
       y_t = Why(h)                                    # logits for this step
3. stack all y_t  →  logits: (batch, seq_len, vocab_size)

Returns: logits, h
```

The line `h = torch.tanh(self.Wxh(x_t) + self.Whh(h))` is the core of the RNN — it is where the recurrence equation is implemented verbatim.

---

#### Cell 5 — Helper Functions

**`get_batch(data, seq_len, batch_size)`**

Samples `batch_size` random overlapping windows from the corpus.

```
For each random start s:
    x[i] = data[s   : s+seq_len]     ← input
    y[i] = data[s+1 : s+seq_len+1]   ← target (shifted by 1)
```

Both tensors have shape `(batch_size, seq_len)`. The one-position shift implements the **language modelling objective**: at every step, predict the next character.

---

**`generate(model, start_char, length=100)`**

Autoregressively generates `length` characters starting from `start_char`.

```
1. seed:   x = [[char_to_idx[start_char]]]   shape (1, 1)
2. loop length times:
       logits, h = model(x, h)
       probs = softmax(logits[0, -1])         # distribution over vocab
       idx   = multinomial(probs, 1)          # sample (stochastic, not argmax)
       result.append(idx_to_char[idx])
       x = [[idx]]                            # feed prediction back as input
3. return ''.join(result)
```

Using `torch.multinomial` (sampling) rather than `argmax` means successive calls give different outputs, producing more varied text.

---

#### Cell 6 — Training Loop

**Hyperparameters**

| Parameter     | Value | Meaning                          |
| ------------- | ----- | -------------------------------- |
| `hidden_size` | 64    | Width of the hidden state vector |
| `seq_len`     | 25    | Characters per training window   |
| `batch_size`  | 16    | Training sequences per update    |
| `lr`          | 0.01  | Adam learning rate               |
| `epochs`      | 2000  | Total gradient steps             |

**Per-epoch procedure**

```
1.  x, y      = get_batch(data, seq_len, batch_size)
2.  logits, _ = model(x)                             # (batch, seq_len, vocab_size)
3.  loss      = cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
4.  optimizer.zero_grad()
5.  loss.backward()                                  # BPTT
6.  clip_grad_norm_(model.parameters(), 5.0)         # exploding gradient guard
7.  optimizer.step()
```

**Loss function**

Cross-entropy measures how much probability mass the model assigns to the correct next character, averaged over all positions:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid x_{\leq i})$$

An untrained model on a ~30-character vocabulary starts near $\ln(30) \approx 3.4$. A well-trained model on this tiny corpus typically reaches below 1.5.

**Why gradient clipping?**

During Backpropagation Through Time (BPTT), gradients flow back through the recurrence by repeatedly multiplying $W_{hh}^T$. If any singular value of $W_{hh}$ exceeds 1, the gradient norm grows **exponentially** with sequence length — the _exploding gradient_ problem. Clipping rescales all gradients proportionally when their global norm exceeds 5.0, preserving direction while bounding magnitude:

$$g \leftarrow g \cdot \frac{5.0}{\|g\|} \quad \text{if } \|g\| > 5.0$$

**Saving weights**

```python
torch.save(model.state_dict(), "handwritten_rnn.pt")
```

Saves only the learned weight tensors. To reload later:

```python
model = VanillaRNN(vocab_size, hidden_size)
model.load_state_dict(torch.load("handwritten_rnn.pt"))
model.eval()
```

---

## Concepts Illustrated

| Concept                                          | Where                           |
| ------------------------------------------------ | ------------------------------- |
| Explicit recurrence loop                         | `VanillaRNN.forward`            |
| Embedding layer (char → vector)                  | `self.embed`                    |
| Hidden state as memory                           | `h` threaded through time steps |
| Language modelling objective (predict next char) | `get_batch`                     |
| Autoregressive generation with sampling          | `generate`                      |
| Exploding gradients & gradient clipping          | training loop                   |
| Cross-entropy loss for sequence models           | training loop                   |
| Saving / loading model weights                   | end of training loop            |

---

## Limitations of Vanilla RNNs

While this implementation clearly shows how RNNs work, vanilla RNNs have known limitations:

- **Vanishing gradients** — information from many steps back fades during BPTT
- **Short effective memory** — the model struggles to capture long-range dependencies
- **Exploding gradients** — mitigated here by gradient clipping, but architecturally fragile

These issues motivated the development of **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit), which use gating mechanisms to control information flow and maintain gradients over longer sequences.

---

## `lstm_version.ipynb` — LSTM Comparison Notebook

This notebook re-implements the same character-level model using PyTorch's built-in `nn.LSTM`, keeping the corpus, batching logic, and every hyperparameter identical to `handwritten_rnn.ipynb`. The only variable that changes is the recurrent architecture itself — this makes the two notebooks directly comparable.

#### Cell 1 — Imports, Data, and Random Seed

Identical to the Vanilla RNN notebook: same corpus, vocabulary, encode/decode helpers, and `torch.manual_seed(42)`.

#### Cell 2 — `LSTMGenerator` Model

```python
class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embed(x)
        out, hidden = self.lstm(embedded, hidden)
        logits = self.fc(out)
        return logits, hidden
```

| Layer   | Type           | Role                                                                                                                                                 |
| ------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embed` | `nn.Embedding` | Same as Vanilla RNN — character index → dense vector                                                                                                 |
| `lstm`  | `nn.LSTM`      | Replaces `Wxh`, `Whh`, and the manual `tanh` loop. Internally runs forget/input/output gates and tracks both a hidden state `h` and a cell state `c` |
| `fc`    | `nn.Linear`    | Same role as `Why` — projects hidden output to vocabulary logits                                                                                     |

**Key architectural differences vs. `VanillaRNN`:**

|                             | `VanillaRNN`                                   | `LSTMGenerator`                                                              |
| --------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| Recurrence                  | Explicit Python `for t in range(seq_len)` loop | Handled internally by `nn.LSTM` — no visible loop                            |
| State carried between steps | Single hidden vector `h`                       | Tuple `(h, c)` — hidden state _and_ cell state                               |
| Per-step computation        | One `tanh(Wxh(x_t) + Whh(h))`                  | Four gates (3× sigmoid, 1× tanh) controlling what to forget, add, and output |
| Layer stacking              | Fixed at 1 layer                               | Configurable via `num_layers`                                                |

Since `nn.LSTM` loops over time internally, the full `embedded` tensor `(batch, seq_len, hidden_size)` is passed in at once — `batch_first=True` tells it to expect `(batch, seq_len, features)` ordering.

#### Cell 3 — Helper Functions

`get_batch` and `generate` are line-for-line identical to the Vanilla RNN notebook, aside from renaming the recurrent state variable to `hidden` (representing the LSTM's `(h, c)` tuple instead of a single tensor).

#### Cell 4 — Training Loop

Uses the same hyperparameters, optimizer, loss function, and gradient clipping as `handwritten_rnn.ipynb` so results are directly comparable. Adds a parameter count at the end:

```python
n_params = sum(p.numel() for p in model.parameters())
print(f"LSTM total params: {n_params}")
```

**Why this matters:** for the same `hidden_size`, `nn.LSTM` has roughly **4×** the parameters of the Vanilla RNN's recurrence weights, because it maintains four internal gates instead of one. This quantifies the capacity trade-off behind LSTM's improved long-range memory.

---

## Vanilla RNN vs. LSTM — Quick Reference

|                         | `handwritten_rnn.ipynb`             | `lstm_version.ipynb`                                            |
| ----------------------- | ----------------------------------- | --------------------------------------------------------------- |
| Recurrence              | Hand-written, explicit              | `nn.LSTM` (built-in)                                            |
| Memory                  | Single hidden state                 | Hidden state + cell state                                       |
| Long-range dependencies | Weak (vanishing gradients)          | Stronger (gated cell state)                                     |
| Parameter count         | Smaller                             | ~4× larger (per hidden unit)                                    |
| Educational purpose     | Shows _how_ an RNN works internally | Shows how the same task is solved with a standard library layer |
