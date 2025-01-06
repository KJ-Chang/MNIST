import torch
import torch.nn as nn
from data import get_dataloader
from models import get_model
from config import *


if __name__ == "__main__":
    _, test_loader = get_dataloader()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 只載入權重
    model_state_dict = torch.load(f'./{MODEL_NAME}.pth', weights_only=True)
    model = get_model(MODEL_NAME)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode():
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_loss += loss
            acc = ((y_pred.argmax(dim=1) == y).sum()) / len(y) * 100
            test_acc += acc
        
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}  , Test Acc: {test_acc:.2f}")



    



