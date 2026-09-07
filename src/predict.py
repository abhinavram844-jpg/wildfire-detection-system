import torch
import sys
from torchvision import models, transforms 
from PIL import Image

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("wildfire_model.pth"))

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(sys.argv[1])
image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim = 1)

predicted_class = output.argmax(dim=1).item()
confidence = probabilities[0][predicted_class].item()

classes = ["no_smoke", "smoke"]
print("Prediction:", classes[predicted_class])
print("Confidence:", f"{confidence * 100: .2f}%")

