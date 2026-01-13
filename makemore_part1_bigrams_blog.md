# Makemore Part 1 (Zero to Hero) — A Text + Code Walkthrough

This is a text-first, notebook-style walk-through of Andrej Karpathy’s *Zero to Hero* video #2: **“The spelled-out intro to language modeling”** (Makemore Part 1). It follows the original video flow closely and keeps all code examples intact, while adding minimal structure so you can run each block in order.

> Goal: build a **character-level bigram language model** for names, first by counting, then by formulating it as a simple neural network trained with gradient descent.

---

## Setup: data and quick inspection

The dataset is `names.txt` (a list of ~32k names, one per line). Load the file and peek at the data.

```python
words = open('names.txt', 'r').read().splitlines()
```

```python
words[:10]
```

```python
len(words)
```

```python
min(len(w) for w in words)
```

```python
max(len(w) for w in words)
```

**Context:** each word is a sequence of characters. A character-level language model predicts the next character given the previous character(s). Here we start with **bigrams** (just the previous character).

---

## Count bigrams directly

Add special start/end tokens and count how often every pair appears.

```python
b = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    b[bigram] = b.get(bigram, 0) + 1
```

Inspect the most common pairs.

```python
sorted(b.items(), key = lambda kv: -kv[1])
```

**Idea:** each word contributes several bigrams — e.g., `emma` gives `(S,e)`, `(e,m)`, `(m,m)`, `(m,a)`, `(a,E)`.

---

## Put counts into a 2D tensor (PyTorch)

A 2D matrix is more convenient than a dictionary. We use `.` as the start/end token.

```python
import torch
```

```python
N = torch.zeros((27, 27), dtype=torch.int32)
```

```python
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
```

```python
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1
```

At this point `N[i, j]` is the count of how often character `i` is followed by `j`.

---

## Visualize the bigram counts

```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
```

**Reading the grid:** row `i` is the current character, column `j` is the next character. The cell value is the bigram count.

---

## Turn counts into probabilities (simple sampling)

```python
N[0]
```

```python
p = N[0].float()
p = p / p.sum()
p
```

```python
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
itos[ix]
```

Sanity-check multinomial sampling:

```python
g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
p
```

```python
torch.multinomial(p, num_samples=100, replacement=True, generator=g)
```

Some quick tensor shape checks (from the video):

Now define the smoothed probability matrix `P` with add-one smoothing.

```python
P = (N+1).float()
P /= P.sum(1, keepdims=True)
```

Shape sanity checks (now that `P` exists):

```python
p.shape
```

```python
P.shape
```

```python
P.sum(1, keepdim=True).shape
```

```python
# 27, 27
# 27,  1
```

```python
P.sum(1).shape
```

```python
# 27, 27
#  1, 27
```

Sample some names from this bigram model:

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```

---

## Likelihood and negative log likelihood (NLL)

We want to **maximize the likelihood** of the observed data under our model. Equivalent objectives:

- maximize log-likelihood (log is monotonic)
- minimize negative log-likelihood
- minimize average negative log-likelihood

```python
# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) + log(b) + log(c)
```

Compute NLL under the bigram model `P`.

```python
log_likelihood = 0.0
n = 0

for w in words:
#for w in ["andrejq"]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')
```

---

## Reframe bigrams as a neural net problem

Create the training set `(x, y)` from bigrams, where `x` is the current character and `y` is the next.

```python
# create the training set of bigrams (x,y)
xs, ys = [], []

for w in words[:1]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    print(ch1, ch2)
    xs.append(ix1)
    ys.append(ix2)
    
xs = torch.tensor(xs)
ys = torch.tensor(ys)
```

```python
xs
```

```python
ys
```

One-hot encode the input indices.

```python
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()
xenc
```

```python
xenc.shape
```

```python
plt.imshow(xenc)
```

```python
xenc.dtype
```

Matrix multiply sanity check:

```python
W = torch.randn((27, 1))
xenc @ W
```

---

## From weights to probabilities (softmax)

```python
logits = xenc @ W # log-counts
counts = logits.exp() # equivalent N
probs = counts / counts.sum(1, keepdims=True)
probs
```

```python
probs[0]
```

```python
probs[0].shape
```

```python
probs[0].sum()
```

```python
# (5, 27) @ (27, 27) -> (5, 27)
```

```python
# SUMMARY ------------------------------>>>>
```

```python
xs
```

```python
ys
```

Now scale up to 27 output neurons (one per character).

```python
# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g)
```

```python
xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
# btw: the last 2 lines here are together called a 'softmax'
```

```python
probs.shape
```

Compute NLL for a few examples to see how it works.

```python
nlls = torch.zeros(5)
for i in range(5):
  # i-th bigram:
  x = xs[i].item() # input character index
  y = ys[i].item() # label character index
  print('--------')
  print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
  print('input to the neural net:', x)
  print('output probabilities from the neural net:', probs[i])
  print('label (actual next character):', y)
  p = probs[i, y]
  print('probability assigned by the net to the the correct character:', p.item())
  logp = torch.log(p)
  print('log likelihood:', logp.item())
  nll = -logp
  print('negative log likelihood:', nll.item())
  nlls[i] = nll

print('=========')
print('average negative log likelihood, i.e. loss =', nlls.mean().item())
```

---

## Optimize the weights with gradient descent

Single mini-step (toy example):

```python
# --------- !!! OPTIMIZATION !!! yay --------------
```

```python
xs
```

```python
ys
```

```python
# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```

```python
# forward pass
xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
loss = -probs[torch.arange(5), ys].log().mean()
```

```python
print(loss.item())
```

```python
# backward pass
W.grad = None # set to zero the gradient
loss.backward()
```

```python
W.data += -0.1 * W.grad
```

---

## Full training on all bigrams

Build the full dataset and train with gradient descent and a small L2 regularization term.

```python
# --------- !!! OPTIMIZATION !!! yay, but this time actually --------------
```

```python
# create the dataset
xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```

```python
# gradient descent
for k in range(1):
  
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  W.data += -50 * W.grad
```

---

## Sample names from the trained network

Compare this sampling block to the earlier count-based model; now we use the learned weights `W` to compute probabilities.

```python
# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```

---

## Conceptual recap

- **Bigram model:** predict the next character using only the previous character.
- **Counts → probabilities:** normalize the count matrix `N` to get `P`.
- **Likelihood:** evaluate model quality via (negative) log-likelihood.
- **Neural net view:** one-hot input → linear layer → softmax → probabilities.
- **Training:** minimize average NLL with gradient descent; add L2 regularization.
- **Sampling:** start from `.` and repeatedly sample the next character until `.` is produced.

---

## Next steps

If you want the *full* notebook flow, run the code blocks above in order. This is a direct text + code reconstruction of the video, with minimal commentary so you can follow the exact progression.
