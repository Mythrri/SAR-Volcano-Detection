import torch
from torchvision import datasets, transforms
from model import get_model

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

data = datasets.ImageFolder('dataset/', transform=transform)
loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

model = get_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    for img, label in loader:
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss {loss.item()}")

torch.save(model.state_dict(), "sar_model.pth")
print("Model saved!")