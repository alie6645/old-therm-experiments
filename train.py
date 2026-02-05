import torch
from torch import nn
from torch.utils.data import DataLoader
from ExperimentData import ExperimentDataset
import torchvision.transforms as transforms
from model import NeuralNetwork

data = ExperimentDataset(
"olddata\\tiny\\rgb",
"olddata\\tiny\\therm",
)

loader = DataLoader(data, batch_size=10, shuffle=True)

for X, y in loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(loader, model, loss_fn, optimizer)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved Model to model.pth")