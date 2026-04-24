# MakeMore Part 3: Activations, Gradients, and Batch Normalization

*Based on Andrej Karpathy's "Neural Networks: Zero to Hero" lecture series*

---

## Introduction

We are continuing our implementation of MakeMore. In the last lecture, we implemented the multilayer perceptron along the lines of Bengio et al. 2003 for character-level language modeling — we took in a few characters in the past and used an MLP to predict the next character in a sequence.

What we'd like to do now is move on to more complex and larger neural networks, like recurrent neural networks and their variations like the GRU, LSTM, and so on. But before we do that, we have to stick around the level of multilayer perceptron for a bit longer, because we need a very good intuitive understanding of the **activations** in the neural net during training, and especially the **gradients** that are flowing backwards — how they behave and what they look like.

This is going to be very important to understand the history of the development of these architectures. We'll see that recurrent neural networks, while they are very expressive (they are a universal approximator and can, in principle, implement all algorithms), they are not very easily optimizable with the first-order gradient-based techniques that we use all the time. The key to understanding why they are not optimizable easily is to understand the activations and the gradients and how they behave during training. A lot of the variants since RNNs have tried to improve that situation.

---

## Starting Code

The starting code for this lecture is largely the code from before, cleaned up a little bit.

### Imports and Data Loading

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

# read in all the words
words = open('names.txt', 'r').read().splitlines()
words[:8]
# There's a total of 32,000 words
len(words)
```

### Character Vocabulary

```python
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)  # 27 characters: 26 lowercase letters + special dot token
```

### Dataset Construction

```python
block_size = 3  # context length: how many characters do we take to predict the next one?

def build_dataset(words):
  X, Y = [], []
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix]  # crop and append
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%
```

We're creating three splits — train, dev, and test — with an 80/10/10 proportion.

### The MLP

This is the identical same MLP from before, except the magic numbers have been pulled out into variables: `n_embd` (dimensionality of the embedding space) and `n_hidden` (number of hidden units in the hidden layer). The same neural net has 11,000 parameters that we optimize over 200,000 steps with a batch size of 32.

The `@torch.no_grad()` decorator is used on the evaluation function — it tells PyTorch that whatever happens in this function will never require any gradients. PyTorch won't do any of the bookkeeping to keep track of gradients in anticipation of an eventual backward pass. It's almost as if all tensors created here have `requires_grad=False`. This makes everything much more efficient because you're telling PyTorch "I will not call `.backward()` on any of this computation and you don't need to maintain the graph under the hood."

At this point, we are starting to get much nicer looking words sampled from the model. Still not amazing and not fully name-like, but much better than the bigram model. The train and val loss are about 2.16.

---

## Problem 1: The Softmax Is Confidently Wrong

The first thing to scrutinize is the initialization. The network is very improperly configured at initialization.

On the 0th iteration, the very first iteration, we are recording a **loss of 27**, and this rapidly comes down to roughly 2. The initialization is all messed up because this is way too high.

### What Loss Should We Expect?

In training of neural nets, it is almost always the case that you will have a rough idea for what loss to expect at initialization, and that just depends on the loss function and the problem setup.

At initialization, there are 27 characters that could come next for any one training example. We have no reason to believe any character to be much more likely than others. So the probability distribution that comes out initially should be a **uniform distribution** assigning about equal probability to all 27 characters:

$$P(\text{any character}) \approx \frac{1}{27}$$

The loss is the negative log probability:

```python
-torch.tensor(1/27.0).log()  # = 3.29
```

So the expected loss at initialization should be about **3.29**, much lower than 27!

### Why the Loss Is So High

What's happening is that at initialization, the neural net is creating probability distributions that are all messed up. Some characters are very confident and some are very not confident. The network is **very confidently wrong**, and that's what makes it record very high loss.

Here's a smaller four-dimensional example of the issue:

```python
# If logits are very close to zero → uniform distribution → expected loss
logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
probs = torch.softmax(logits, dim=0)  # [0.25, 0.25, 0.25, 0.25]
loss = -probs[2].log()  # 1.38 (expected loss for 4 classes)

# But if logits take extreme values → confidently wrong → very high loss
logits = torch.randn(4) * 10  # extreme values
# Now likely to record very high loss because we're confidently wrong
```

The key insight: **we want the logits to be roughly zero** (or at least equal) when the network is initialized. By symmetry, we just want all zeros and record the loss we expect at initialization.

### The Fix

Looking at how logits are calculated: `logits = h @ W2 + b2`

1. **Zero the bias**: `b2` is initialized as random values, but we want roughly zero, so multiply by 0: `b2 = torch.randn(vocab_size) * 0`

2. **Scale down W2**: If we want logits to be very small, we multiply W2 by a small number: `W2 = torch.randn((n_hidden, vocab_size)) * 0.01`

With `W2 * 0.01`, the initial loss drops to about 3.32 — close to our expected 3.29.

**Why not set W2 to exactly zero?** You usually want it to be small numbers instead of exactly zero. Setting weights exactly to zero breaks symmetry in harmful ways (more on this below). With 0.01, the loss is close enough but has some entropy used for symmetry breaking.

### Impact on Training

With this fix:
- The loss no longer has a "hockey stick" appearance — the first few iterations were previously just squashing down the logits (easy work), and now we're spending those cycles on the actual hard gains of training
- Validation loss improved from **2.16 → 2.13**
- The improvement comes from not wasting optimization cycles squashing down weights that are way too high at initialization

---

## Problem 2: Tanh Layer Is Too Saturated

Even though the loss looks good now, there's a deeper problem lurking inside the neural net's initialization.

### Diagnosing the Problem

Looking at the hidden activations `h`, many elements are exactly 1 or -1. Recall that `torch.tanh` is a squashing function that takes arbitrary numbers and squashes them into the range [-1, 1].

```python
# Histogram of h reveals most values are at -1 or +1
plt.hist(h.view(-1).tolist(), 50)

# The pre-activations that feed into tanh are too extreme:
plt.hist(hpreact.view(-1).tolist(), 50)
# Distribution spans from -15 to +15!
```

### Why Saturated Tanh Is Dangerous

During backpropagation, we propagate through `torch.tanh`. From the micrograd implementation:

```python
# In tanh backward pass:
# t = tanh(x), where t is between -1 and 1
# local gradient = (1 - t**2)
out.grad = (1 - t**2) * upstream_grad
```

**If `t ≈ 1` or `t ≈ -1`**: `(1 - t²) ≈ 0`, which **kills the gradient**. No matter what the upstream gradient is, we are stopping backpropagation through this tanh unit.

**If `t ≈ 0`**: `(1 - t²) ≈ 1`, and the gradient passes through unchanged.

Intuitively: if a tanh neuron's output is very close to 1, we are in the flat tail of the tanh. Changing the input doesn't impact the output much, so there's no influence on the loss, and indeed the gradient vanishes.

The gradient flowing through tanh can only ever **decrease**. The amount it decreases is proportional to how far you are in the flat tails.

### Dead Neurons

We can check which neurons are saturated:

```python
# Boolean tensor: True where |h| > 0.99 (in flat tails)
(h.abs() > 0.99).float()
# Visualize as image: white = saturated, black = active
```

We would be in a lot of trouble if an entire **column** were white — that would be a **dead neuron**. A dead neuron is one where no single example ever activates this tanh in the active part. If all examples land in the tail, this neuron will never learn.

This is not unique to tanh:
- **Sigmoid**: Same issue — it's a squashing function with flat tails
- **ReLU**: Has a completely flat region below zero. If a ReLU neuron never activates (pre-activation is always negative), it's dead forever. This is "permanent brain damage" in the network
  - Can happen at initialization (by chance)
  - Can happen during optimization (too high learning rate knocks neuron off the data manifold)
- **Leaky ReLU**: Does not suffer from this issue because it doesn't have flat tails — you almost always get gradients
- **ELU**: May suffer from this issue because it has flat parts on the negative side

### The Fix

The pre-activations `hpreact = emb @ W1 + b1` are too far from 0. We need them closer to 0:

```python
b1 = torch.randn(n_hidden) * 0.01   # small bias for diversity
W1 = torch.randn((n_embd * block_size, n_hidden)) * 0.2  # scale down
```

With `W1 * 0.2`, the histogram of h shows a much better distribution, and the saturation map shows essentially no white (no saturated neurons above 0.99).

**Result**: Validation loss improved from **2.13 → 2.10**.

This matters much more for deeper networks. With a shallow one-layer MLP, the optimization is quite forgiving — even with terrible initialization, the network eventually learns. But with 50-layer networks, these problems stack up, and you can reach a place where the network simply doesn't train at all.

---

## Kaiming Initialization

We've been using magic numbers like 0.2 to scale our weights. Where do these come from? How do we set these scales for large neural networks with many layers? Obviously no one does this by hand. There are relatively principled ways.

### The Core Problem

Consider random input `x` drawn from a Gaussian (1000 examples, 10-dimensional) and a weight matrix `W` (10×200, also Gaussian):

```python
x = torch.randn(1000, 10)
w = torch.randn(10, 200)
y = x @ w
```

The input has mean 0 and std 1. After multiplication, the mean stays 0 (symmetric operation), but the **standard deviation grows** — from 1 to about 3. The Gaussian is expanding!

We don't want that. We want most of the neural net to have relatively similar activations — roughly unit Gaussian throughout the network.

- If we scale `w` by a larger number (e.g., `* 5`), std grows to ~15
- If we scale `w` down (e.g., `* 0.2`), std shrinks to ~0.6

**The correct answer**: divide by the **square root of the fan-in**:

```python
w = torch.randn(10, 200) / 10**0.5  # fan_in = 10
y = x @ w
# y now has std ≈ 1.0 — distribution preserved!
```

### The Kaiming He et al. Paper

The paper ["Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852) (Kaiming He et al.) studied this in detail for convolutional neural networks with ReLU and PReLU nonlinearities. The analysis is very similar to what we've derived:

For **ReLU** nonlinearity: because you're throwing away half the distribution (clamping negatives to 0), they find you need a compensating gain. Their formula:

$$\text{std}(W) = \sqrt{\frac{2}{\text{fan\_in}}}$$

The factor of 2 compensates for ReLU discarding half the distribution.

### Kaiming Init in PyTorch

In `torch.nn.init`, you'll find `kaiming_normal_`:

- **mode**: `fan_in` (normalize activations) or `fan_out` (normalize gradients). The paper found this doesn't matter too much; most people leave the default (`fan_in`)
- **nonlinearity**: Determines the gain:
  - `'linear'` (identity): gain = 1
  - `'relu'`: gain = √2
  - `'tanh'`: gain = 5/3

The gain compensates for the **contractive** nature of nonlinearities. Tanh squashes the distribution (squeezes the tails), so you need to boost the weights slightly to renormalize everything back to unit standard deviation.

### Applying Kaiming Init to Our Network

The standard deviation we want is: `gain / sqrt(fan_in)`

For our network with tanh:

```python
W1 = torch.randn((n_embd * block_size, n_hidden)) * (5/3) / ((n_embd * block_size)**0.5)
# fan_in = n_embd * block_size = 30
# (5/3) / sqrt(30) ≈ 0.3
```

This gives us 0.3 instead of our hand-tuned 0.2. The result: validation loss is about the same (2.10), but we arrived there **without magic numbers** — we have something semi-principled that will scale to much bigger networks.

### PyTorch's Default Initialization

PyTorch's `nn.Linear` layer initializes weights using:

$$W \sim \text{Uniform}\left(-\frac{1}{\sqrt{k}}, \frac{1}{\sqrt{k}}\right)$$

where $k$ is the fan-in. Same motivation — ensuring roughly Gaussian output from a roughly Gaussian input. They use a uniform distribution instead of Gaussian, and gain = 1 (no nonlinearity adjustment), but it's the same idea.

### Why Precise Initialization Is Less Critical Today

A number of modern innovations have made everything significantly more stable:

1. **Residual connections** (covered later)
2. **Normalization layers**: batch normalization, layer normalization, group normalization
3. **Better optimizers**: RMSProp and especially **Adam** (not just simple SGD)

All of these make it less important to precisely calibrate initialization. But in practice, normalizing weights by `1/sqrt(fan_in)` is still the standard starting point.

---

## Batch Normalization

Batch normalization came out in 2015 from a team at Google and was an extremely impactful paper because it made it possible to train very deep neural nets quite reliably. It basically just worked.

### The Core Idea

We have hidden states `hpreact` and we don't want them to be way too small (tanh inactive) or too large (tanh saturated). We want them to be roughly Gaussian — zero mean and unit standard deviation, at least at initialization.

**The insight**: If you want roughly Gaussian activations, why not just take them and **normalize them to be Gaussian**? It sounds kind of crazy, but you can just do that because standardizing hidden states is a perfectly differentiable operation.

### Implementation

`hpreact` has shape `[32, 200]` (32 examples × 200 neurons). We compute statistics **across the batch dimension** (dimension 0):

```python
# Calculate batch statistics
bnmeani = hpreact.mean(0, keepdim=True)   # shape: [1, 200]
bnstdi = hpreact.std(0, keepdim=True)     # shape: [1, 200]

# Normalize: every neuron's firing rate becomes unit Gaussian over the batch
hpreact = (hpreact - bnmeani) / bnstdi
```

This is called **batch** normalization because we normalize over the batch. The mean and std are computed per-neuron across all examples in the batch.

### Scale and Shift (Gain and Bias)

We want activations to be roughly Gaussian **at initialization**, but we don't want them **forced** to be Gaussian always. We'd like the neural net to move this distribution around — make some tanh neurons more trigger-happy or less trigger-happy, make the distribution more diffuse or more sharp.

So we add learnable **scale** (gain) and **shift** (bias) parameters:

```python
# BatchNorm parameters
bngain = torch.ones((1, n_hidden))    # initialized to 1
bnbias = torch.zeros((1, n_hidden))   # initialized to 0

# Apply batch norm
hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
```

At initialization (gain=1, bias=0), each neuron's firing values are exactly unit Gaussian. During optimization, backpropagation can adjust gain and bias so the network has full ability to do whatever it wants internally.

### Full Training Loop with Batch Normalization

```python
n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
W2 = torch.randn((n_hidden, vocab_size),           generator=g) * 0.01
b2 = torch.randn(vocab_size,                       generator=g) * 0

# BatchNorm parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]
for p in parameters:
  p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix]

  # forward pass
  emb = C[Xb]
  embcat = emb.view(emb.shape[0], -1)
  # Linear layer
  hpreact = embcat @ W1  # no bias! (BatchNorm makes it redundant)
  # BatchNorm layer
  bnmeani = hpreact.mean(0, keepdim=True)
  bnstdi = hpreact.std(0, keepdim=True)
  hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
  with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
  # Non-linearity
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Yb)

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  if i % 10000 == 0:
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
```

### Loss Log — Tracking Improvements

| Change | Train Loss | Val Loss |
|--------|-----------|----------|
| Original | 2.1245 | 2.1682 |
| Fix softmax confidently wrong | 2.07 | 2.13 |
| Fix tanh layer too saturated at init | 2.0356 | 2.1027 |
| Use semi-principled Kaiming init | 2.0377 | 2.1070 |
| Add batch norm layer | 2.0668 | 2.1048 |

The batch norm result is comparable to our careful initialization. This is expected — with a simple one-hidden-layer network, we could calculate exactly what the scale of W should be. But for much deeper networks with lots of different types of operations, it becomes intractable to tune all weight scales manually. BatchNorm layers can simply be sprinkled throughout.

### The Coupling Problem

Batch normalization introduces something terribly strange and unnatural. It used to be that a single example feeds into the neural net and produces activations deterministically. But now, because of normalization through the batch, **examples are coupled mathematically** in the forward and backward pass.

The hidden state activations and logits for any one input example are not just a function of that example — they're also a function of all the other examples that happen to come for a ride in that batch. If you imagine sampling different batches, the activations will **jitter** because the batch statistics change.

Surprisingly, this turns out to be **good** as a side effect — it acts as a **regularizer**. The jittering pads out input examples, introduces entropy, and makes it harder for the neural net to overfit to specific examples. It's a form of implicit data augmentation.

This regularizing effect has actually made it **harder to remove batch normalization** from practice. People have tried to deprecate it in favor of other normalization techniques that don't couple examples (layer normalization, instance normalization, group normalization), but batch norm works well partly because of this regularizing property.

### Inference with Batch Normalization

At inference time, we want to feed in a single example and get a prediction. But the neural net now expects batches (to compute mean and std). How do we handle this?

**Option 1: Post-training calibration** — After training, pass the entire training set through and compute the mean and std once:

```python
# calibrate the batch norm at the end of training
with torch.no_grad():
  emb = C[Xtr]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1
  bnmean = hpreact.mean(0, keepdim=True)
  bnstd = hpreact.std(0, keepdim=True)
```

Then use these fixed values at test time instead of per-batch statistics.

**Option 2: Running estimates** (preferred) — Keep a running mean of the mean and std during training, updated on the side with an exponential moving average:

```python
with torch.no_grad():
  bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
  bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
```

This is how PyTorch implements it — no need for a second calibration stage. During inference, the running mean/std are used:

```python
# At test time: use running statistics, not batch statistics
@torch.no_grad()
def split_loss(split):
  x, y = {'train': (Xtr, Ytr), 'val': (Xdev, Ydev), 'test': (Xte, Yte)}[split]
  emb = C[x]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1
  hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())
```

### Practical Details

**Epsilon**: The batch norm formula includes a small `ε` (default 1e-5) to prevent division by zero when variance is exactly zero.

**Bias in preceding layers is redundant**: When using batch normalization after a linear layer, the bias of that linear layer is useless — it gets subtracted out by the batch norm's mean calculation:

```python
# With batch norm, this bias is wasteful:
hpreact = embcat @ W1 + b1  # b1 gets subtracted by batchnorm!

# Better: disable bias
hpreact = embcat @ W1  # no bias
# The batch norm's own bias (bnbias) handles the offset
```

This is why in PyTorch's ResNet implementation, you'll see `bias=False` on every convolution layer that's followed by batch norm:

```python
self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False)
self.bn1 = nn.BatchNorm2d(out_channels)
```

### Batch Normalization Summary

A batch normalization layer:

1. **Parameters** (trained with backpropagation): gain (γ) and bias (β)
2. **Buffers** (trained with running mean update, NOT backpropagation): running mean and running standard deviation

What it does:
1. Calculate the mean and std of activations over the batch
2. Center the batch to be unit Gaussian
3. Scale and shift by the learned gain and bias
4. Keep track of running mean and std via exponential moving average for use at inference

The typical motif in deep networks: **Weight layer → Batch normalization → Nonlinearity** (repeated many times).

---

## PyTorchifying the Code: Deeper Networks

Now let's restructure our code to look much more like what you'd encounter in PyTorch, organizing into modules.

### The Linear Layer

```python
class Linear:

  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None

  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])
```

This mirrors PyTorch's `nn.Linear`:
- Takes `fan_in`, `fan_out`, and optional `bias`
- Initializes weight using Kaiming init: `/ fan_in**0.5`
- Bias defaults to zeros
- Forward pass computes `Wx + b`

### The BatchNorm1d Layer

```python
class BatchNorm1d:

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)

  def __call__(self, x):
    if self.training:
      xmean = x.mean(0, keepdim=True)
      xvar = x.var(0, keepdim=True)
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
```

Key points:
- `.training` attribute controls behavior (train mode vs eval mode) — same as PyTorch
- During training: uses batch statistics and updates running estimates
- During eval: uses running mean and variance (fixed)
- `torch.no_grad()` context manager prevents PyTorch from building a computational graph for the buffer updates
- Only gamma and beta are returned as parameters (running stats are NOT part of gradient-based optimization)

### The Tanh Layer

```python
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []
```

### Building a Deeper Network

```python
n_embd = 10
n_hidden = 100
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((vocab_size, n_embd), generator=g)
layers = [
  Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]

with torch.no_grad():
  # last layer: make less confident
  layers[-1].gamma *= 0.1
  # all other layers: apply gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 5/3  # tanh gain

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))  # 46,000 parameters
for p in parameters:
  p.requires_grad = True
```

Note:
- 5 hidden layers of 100 neurons each
- Every `Linear` uses `bias=False` because `BatchNorm1d` follows
- The last BatchNorm's gamma is scaled by 0.1 to make the initial softmax less confident
- The `5/3` gain compensates for tanh's contractive nature

### Training Loop

```python
max_steps = 200000
batch_size = 32
lossi = []
ud = []  # update-to-data ratio tracker

for i in range(max_steps):
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix]

  # forward pass
  emb = C[Xb]
  x = emb.view(emb.shape[0], -1)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Yb)

  # backward pass
  for layer in layers:
    layer.out.retain_grad()
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i < 150000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  if i % 10000 == 0:
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  with torch.no_grad():
    ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])
```

### Evaluation and Sampling

```python
# put layers into eval mode (use running stats instead of batch stats)
for layer in layers:
  layer.training = False

split_loss('train')
split_loss('val')

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
      emb = C[torch.tensor([context])]
      x = emb.view(emb.shape[0], -1)
      for layer in layers:
        x = layer(x)
      logits = x
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    print(''.join(itos[i] for i in out))
```

---

## Diagnostic Tools for Neural Network Training

Three types of visualizations are essential for understanding whether your neural network is in a good state.

### 1. Forward Pass Activation Distributions

```python
plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]):
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' %
      (i, layer.__class__.__name__, t.mean(), t.std(),
       (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends)
plt.title('activation distribution')
```

What to look for:
- All layers should have roughly the **same standard deviation** (~0.65 for tanh with proper gain)
- **Saturation** should be low (~2-5%) — not too many values at ±1
- Distributions should be **homogeneous** across layers

**If gain is too small (e.g., 1.0 instead of 5/3)**: activations shrink towards 0 in deeper layers, saturation drops to 0%.

**If gain is too large (e.g., 3.0)**: saturations become way too large (>20%), killing gradients.

### 2. Backward Pass Gradient Distributions

```python
plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]):
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' %
      (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends)
plt.title('gradient distribution')
```

What to look for:
- All layers should have roughly **equal gradient magnitudes**
- No shrinking to zero or exploding to infinity
- **Asymmetry** between layers = trouble

### 3. Weight Gradient and Update-to-Data Ratios

```python
plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' %
      (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution')
```

The most informative plot tracks the **update-to-data ratio** over time:

```python
plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k')  # target line at 1e-3
plt.legend(legends)
```

The update-to-data ratio measures: `std(lr * gradient) / std(parameter_values)`. This tells you how much the parameters are actually changing at each step.

**The golden rule: this ratio should be around 1e-3 (i.e., -3 on the log10 scale).**

- **Much above -3** (e.g., -1): learning rate too high, parameters changing too aggressively
- **Much below -3** (e.g., -5): learning rate too low, parameters barely changing

### Why Linear-Only Networks Are Uninstructive

If you remove all tanh nonlinearities, you get a giant linear sandwich. A stack of linear layers collapses to a **single linear layer** in terms of representation power — `f(x) = W₁W₂W₃...x + b` is still just `f(x) = Wx + b`.

However, the optimization dynamics differ: the backward pass chain rule creates interesting dynamics, and there are entire papers analyzing infinitely layered linear networks.

The tanh nonlinearities are what allow the sandwich to approximate **any arbitrary function** (universal approximation theorem).

### Effect of BatchNorm on Robustness

With batch normalization layers in the sandwich:

**Activations are guaranteed to look good** — before every tanh, there's a normalization. The standard deviation stabilizes at ~0.65, saturation at ~2%, and everything is homogeneous across layers.

**Robustness to gain changes**: Even if the gain is set to 0.2 (much too low without batch norm), the activations are exactly unaffected because of the explicit normalization. The forward and backward passes look OK.

**However**: the update-to-data ratios are affected by gain changes. If gains are too large, the updates come out lower. You may need to retune the learning rate. So batch norm doesn't give a completely free pass — but it makes things significantly more robust.

**Even without fan-in normalization**: With batch norm, using raw Gaussian weights (no `1/sqrt(fan_in)` scaling) still produces well-behaved forward and backward passes. You'd just need to adjust the learning rate.

---

## Summary

### What We Covered

1. **Understanding activations and gradients** — The importance of monitoring them, especially as networks get bigger and deeper

2. **Initialization matters** — Three progressive fixes:
   - Fixing the output layer to not be confidently wrong (loss 2.17 → 2.13)
   - Fixing tanh saturation by scaling weights down (2.13 → 2.10)
   - Using Kaiming initialization for a principled approach (same result, no magic numbers)

3. **Batch normalization** — A layer that explicitly normalizes activations to be unit Gaussian:
   - Introduced in 2015, extremely impactful
   - Has learnable gain/bias parameters
   - Maintains running statistics for inference
   - Couples examples in a batch (undesirable but gives regularization)
   - No one likes it, but it works well; alternatives: layer norm, group norm, instance norm

4. **PyTorchifying code** — Linear, BatchNorm1d, and Tanh as reusable modules, matching PyTorch's API

5. **Diagnostic tools**:
   - Forward activation histograms (check for saturation)
   - Backward gradient histograms (check for vanishing/exploding)
   - Update-to-data ratios (should be ~1e-3)

### What We Didn't Cover

- Performance wasn't the goal — with batch norm on a deeper network, results were similar because the bottleneck is the **context length** (only 3 characters), not optimization
- Full mathematical explanation of how gain changes affect the backward pass through batch norm
- The field of initialization is still an active area of research — we haven't solved it

### What's Next

Recurrent neural networks — which are effectively very deep networks (you unroll the loop during optimization). That's where all of this analysis around activation statistics and normalization layers will become very, very important for good performance.

---

## Appendix: Bonus Code from the Notebook

### Interactive BatchNorm Visualization Widget

```python
from ipywidgets import interact
import scipy.stats as stats
import numpy as np

def normshow(x0):
  g = torch.Generator().manual_seed(2147483647+1)
  x = torch.randn(5, generator=g) * 5
  x[0] = x0
  mu = x.mean()
  sig = x.std()
  y = (x - mu)/sig

  plt.figure(figsize=(10, 5))
  plt.plot([-6,6], [0,0], 'k')
  xx = np.linspace(-6, 6, 100)
  plt.plot(xx, stats.norm.pdf(xx, mu, sig), 'b')  # input distribution
  plt.plot(xx, stats.norm.pdf(xx, 0, 1), 'r')     # output distribution
  for i in range(len(x)):
    plt.plot([x[i],y[i]], [1, 0], 'k', alpha=0.2)
  plt.scatter(x.data, torch.ones_like(x).data, c='b', s=100)
  plt.scatter(y.data, torch.zeros_like(y).data, c='r', s=100)
  plt.xlim(-6, 6)
  plt.title('input mu %.2f std %.2f' % (mu, sig))

interact(normshow, x0=(-30,30,0.5))
```

### Linear Layer: Forward and Backward Statistics

```python
g = torch.Generator().manual_seed(2147483647)
a = torch.randn((1000,1), requires_grad=True, generator=g)
b = torch.randn((1000,1000), requires_grad=True, generator=g)
c = b @ a
loss = torch.randn(1000, generator=g) @ c
a.retain_grad(); b.retain_grad(); c.retain_grad()
loss.backward()

print('a std:', a.std().item())   # ~1.0
print('b std:', b.std().item())   # ~1.0
print('c std:', c.std().item())   # ~31.6 (expanded by sqrt(1000))
print('c grad std:', c.grad.std().item())
print('a grad std:', a.grad.std().item())
print('b grad std:', b.grad.std().item())
```

### Linear + BatchNorm: Statistics

```python
g = torch.Generator().manual_seed(2147483647)
n = 1000
inp = torch.randn(n, requires_grad=True, generator=g)
w = torch.randn((n, n), requires_grad=True, generator=g)
x = w @ inp
# BatchNorm
xmean = x.mean()
xvar = x.var()
out = (x - xmean) / torch.sqrt(xvar + 1e-5)
loss = out @ torch.randn(n, generator=g)
inp.retain_grad(); x.retain_grad(); w.retain_grad(); out.retain_grad()
loss.backward()

print('inp std:', inp.std().item())       # ~1.0
print('x std:', x.std().item())           # ~31.6 (before normalization)
print('out std:', out.std().item())       # ~1.0 (after normalization!)
print('out grad std:', out.grad.std().item())
print('x grad std:', x.grad.std().item())
print('w grad std:', w.grad.std().item())
print('inp grad std:', inp.grad.std().item())
```

This demonstrates batch normalization's key property: no matter how large the pre-normalization activations grow (`x std ≈ 31.6`), the output is always unit Gaussian (`out std ≈ 1.0`).
