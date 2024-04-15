import torch
from torch.utils.data import DataLoader

from data import Stenosis_Dataset
from UNet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0], device=device))
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

train_set = Stenosis_Dataset(mode="train")
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
val_set = Stenosis_Dataset(mode="val")
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, drop_last=False)

for epoch in range(10):
    print(f"EPOCH: {epoch}/10")
    model.train()
    for batch_idx, (inputs, masks) in enumerate(train_loader):
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        pred_masks = model(inputs)
        loss = criterion(pred_masks, masks)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"batch_idx: {batch_idx}, train_loss: {loss.item()}")
    lr_scheduler.step()
    
    model.eval()
    with torch.no_grad():
        total_samples, total_f1_score = 0, 0
        for batch_idx, (inputs, masks) in enumerate(val_loader):
            inputs, masks = inputs.to(device), masks.to(device)
            pred_masks = model(inputs)
            pred_masks = pred_masks[:, 1, :, :] > pred_masks[:, 0, :, :]
            tp = (masks * pred_masks).sum()
            fp = ((1 - masks) * pred_masks).sum()
            fn = (masks * ~pred_masks).sum()
            f1 = tp / (tp + 0.5 * (fp + fn))
            total_samples += inputs.size(0)
            total_f1_score += f1 * inputs.size(0)
        print(f"val_f1_score: {total_f1_score / total_samples}")
