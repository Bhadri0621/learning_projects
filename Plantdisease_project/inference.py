import torch.nn as nn
import torch.optim as optim
from main import CNNModel
import torch
import torch.nn.functional as f
from classes import classes
model=CNNModel()
device='cpu'
model.load_state_dict(
    torch.load(
    'best_model.pt',
    map_location=device
    )
)
model.eval()
def Prediction(x):
    with torch.no_grad():
        
        predictions=model(x)
        probabilities=f.softmax(predictions,dim=1)

        confidence,predicted_class=torch.max(probabilities,dim=1)

        conf_percent=(f'{confidence.item()*100:.2f}%')

        class_name=classes[
            predicted_class.item()
        ]

    return (conf_percent,class_name )