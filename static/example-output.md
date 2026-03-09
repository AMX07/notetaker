# introduction to neural networks

## what is a neural network?

So today I want to talk about neural networks. A neural network is essentially a computational model inspired by the way biological neurons work in the human brain. You've got these interconnected nodes, and each connection has a weight associated with it.

The basic idea is pretty simple. You take some input, multiply it by weights, add a bias, and then pass it through an activation function. That's it. That's a single neuron.

## layers and architecture

When we stack these neurons together, we get layers. A typical neural network has three types of layers:

- **Input layer** — this is where your data comes in
- **Hidden layers** — this is where the computation happens
- **Output layer** — this gives you your prediction

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

## the training process

Training a neural network comes down to three steps that we repeat over and over:

1. **Forward pass** — feed data through the network, get a prediction
2. **Compute loss** — compare the prediction to the actual answer
3. **Backpropagation** — adjust the weights to reduce the error

The key insight is that we use gradient descent. We compute the gradient of the loss with respect to each weight, and then we nudge each weight in the direction that reduces the loss. The learning rate controls how big those nudges are.

$$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$$

If the learning rate is too high, you'll overshoot. Too low, and training takes forever.

## summary

Neural networks learn by adjusting weights through backpropagation. The architecture — how many layers, how many neurons — depends on your problem. Start simple, add complexity only when needed.
