import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader

data_dir = "data/chestxray"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    weights.transforms().normalize  
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    weights.transforms().normalize
])

train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
val_ds   = datasets.ImageFolder(f"{data_dir}/val",   transform=val_tf)
test_ds  = datasets.ImageFolder(f"{data_dir}/test",  transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)

model = efficientnet_b0(weights=weights)     

n_classes = len(train_ds.classes)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, n_classes)  # înlocuiește head
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  

def run_epoch(loader, train=False):
    model.train(train)
    total, correct, total_loss = 0, 0, 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(x)
            loss = criterion(out, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item()*x.size(0)
        correct += (out.argmax(1)==y).sum().item()
        total += x.size(0)
    return total_loss/total, correct/total

for epoch in range(1, 11):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    print(f"Ep {epoch:02d} | train {tr_acc:.3f} | val {val_acc:.3f}")


_, test_acc = run_epoch(test_loader, train=False)
print("Test acc:", round(test_acc,3))
