import torch
import torch.nn as nn
import torch.optim as optim
from models import get_model
from config import *
from tqdm import tqdm
from data import get_dataloader


if __name__ == "__main__":
    train_loader, _ = get_dataloader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(MODEL_NAME)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in tqdm(range(EPOCH)):
        print(f'\nEpoch: {epoch+1}')
        train_loss = 0.0
        train_acc = 0.0
        for (x, y) in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            model.train()

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            acc = ((y_pred.argmax(dim=1) == y).sum()) / len(y) * 100

            train_loss += loss
            train_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print(f"Train Loss: {train_loss:.4f}  , Train Acc: {train_acc:.2f}")

    # save model
    torch.save(model.state_dict(), f'{MODEL_NAME}.pth')

