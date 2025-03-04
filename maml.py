import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


# Model: Simple neural network
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.layers(x)

# MAML Training Setup
model = Regressor()
meta_optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Simulate tasks (e.g., different sine functions)
def create_task():
    amplitude = torch.rand(1) * 5  # Random amplitude between 0 and 5
    phase = torch.rand(1) * 2 * torch.pi  # Random phase

    x = torch.rand(500, 1) * 10  # 5 support points
    print('x shape',x.shape)
    x1 = torch.rand(500, 1) * 10  # 5 support points
    y = amplitude * torch.sin(x1 + phase)
    print('y shape', y.shape)
    # print(x)
    # print(y)
    # fig = plt.figure()
    # plt.scatter(x,y)
    # plt.show()
    return x, y

# MAML Training Loop
for step in range(1000):
    # Sample a task
    x_support, y_support = create_task()
    x_query, y_query = create_task()  # Query set from same task

    # Inner loop: Adapt with support set
    fast_weights = dict(model.named_parameters())
    for i in range(5):  # 5 gradient steps
        pred = model(x_support)
        loss = loss_fn(pred, y_support)
        print(i)
        grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True, allow_unused=True)
        print(type(fast_weights.items()))
        print(type(grads))
        print(len(grads))

        fast_weights = {name: param - 0.01 * grad for (name, param), grad in zip(fast_weights.items(), grads)}
        print(type(fast_weights.values()))

    # Outer loop: Evaluate on query set
    pred_query = model(x_query)
    meta_loss = loss_fn(pred_query, y_query)
    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()