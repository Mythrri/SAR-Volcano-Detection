import torch
from PIL import Image
from torchvision import transforms
from model import get_model
import os, random

model = get_model()
model.load_state_dict(torch.load("sar_model.pth", map_location="cpu"))
model.eval()

# Get images
all_images = []
for root, dirs, files in os.walk("dataset"):
    for f in files:
        if f.lower().endswith((".jpg",".png",".jpeg")):
            all_images.append(os.path.join(root,f))

img_path = random.choice(all_images)
print(f"\nTesting Image: {img_path}")

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
img = Image.open(img_path).convert("RGB")
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    pred = output.argmax().item()

# 0 = deformation, 1 = non_deformation (alphabetical order)
print("\n===== FINAL OUTPUT =====")
if pred == 0:
    print("🌋 RESULT: VOLCANO DEFORMATION DETECTED!")
else:
    print("✅ RESULT: NO DEFORMATION (Normal)")

print(f"Image from folder: {os.path.basename(os.path.dirname(img_path))}")
print("========================")