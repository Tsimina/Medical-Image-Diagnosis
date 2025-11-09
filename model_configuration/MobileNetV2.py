import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

data_dir = "data/chestxray"   # organize: train/val/test subfolders

# transforms
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
val_ds   = datasets.ImageFolder(f"{data_dir}/val",   transform=val_tf)
test_ds  = datasets.ImageFolder(f"{data_dir}/test",  transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=True)
num_ftrs = model.classifier[1].in_features
n_classes = len(train_ds.classes)
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, n_classes))
model = model.to(device)

# loss + opt
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# training loop (simplified)
def train_epoch():
    model.train()
    total_loss = 0
    correct = 0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred==y).sum().item()
    return total_loss/len(train_ds), correct/len(train_ds)

def eval_loader(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred==y).sum().item()
    return correct / (len(loader.dataset))

# run training
for epoch in range(1, 11):
    tr_loss, tr_acc = train_epoch()
    val_acc = eval_loader(val_loader)
    print(f"Epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")
