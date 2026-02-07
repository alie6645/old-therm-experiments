import torch
from torch.utils.data import DataLoader
from model import NeuralNetwork
from ExperimentData import ExperimentDataset
import cv2

test_data = ExperimentDataset(
"olddata\\tiny\\rgb",
"olddata\\tiny\\therm",
len=100
)

loader = DataLoader(test_data, batch_size=10, shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

torch.no_grad()
model.eval()

rgb, therm = next(iter(loader))
pred = model(rgb)
pred = torch.reshape(pred, (10, 1, 32, 32))
therm = therm.squeeze()
pred = pred.squeeze()

cv2.imshow("rgb", rgb[0][0].numpy())
cv2.waitKey(0)
cv2.imshow("therm", therm[0].numpy())
cv2.waitKey(0)
cv2.imshow("pred", pred[0].detach().numpy())
cv2.waitKey(0)
print(therm[0])
print(pred[0])

