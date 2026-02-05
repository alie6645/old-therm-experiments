from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from ExperimentData import ExperimentDataset

data = ExperimentDataset(
"olddata\\tiny\\rgb",
"olddata\\tiny\\therm"
)

loader = DataLoader(data, batch_size=10, shuffle=True)

rgb, therm = next(iter(loader))
rgb = rgb.squeeze()
therm = therm.squeeze()
cv2.imshow("rgb", rgb[0][0].numpy())
cv2.waitKey(0)
cv2.imshow("therm", therm[0][0].numpy())
cv2.waitKey(0)
