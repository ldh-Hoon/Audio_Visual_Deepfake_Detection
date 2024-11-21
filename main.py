
import os, importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.LAV_small import *
from dataset.FakeAVCeleb import FakeAVCeleb

from train import *
from model import *

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #root_dir = 'C:/Users/User/Downloads/LAV-small'
    root_dir = 'C:/Users/User/Downloads/FakeAVCeleb_v1.2'

    train_dataset = FakeAVCeleb(root_dir=root_dir, split='train', num_samples=1000)
    val_dataset = FakeAVCeleb(root_dir=root_dir, split='val', num_samples=1000)
    test_dataset = FakeAVCeleb(root_dir=root_dir, split='test', num_samples=1000)

    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=test_dataset.collate_fn)

    for video, audio, label in train_loader:
        print(f"Train Video shape: {video.shape}, {audio.shape}, Label: {label}")
        break

    num_epochs = 5
    learning_rate = 0.0001
    models = [
                #"CNN_LSTM_each_AV", 
                #"CNN_LSTM_AV",
                #"CNN_LSTM_cross_attention_AV", 
                #"CNN_BiMAMBA_each_AV", 
                #"CNN_BiMAMBA_MHA_AV",
                #"CNN_Transformer_AV",
                #"MeseInception_AV",
                #"Resnet_LSTM_AV",
                "KAN_CNN_LSTM_each_AV",
                #"KAN_CNN_Transformer_AV",
                #"AVT_DWF_AV",
                #"CNN_LSTM_V",
                #"AVT_MAMBA_AV",
                #"CNN_BiMAMBA_AV"
            ]

    save_dict = "./trained_FakeAVCeleb"
    criterion = nn.CrossEntropyLoss()
    log = dict()

    for model_name in models:
        print(model_name)
        module = importlib.import_module(f"model.{model_name}")
        model_class = getattr(module, model_name)
        model = model_class(num_classes=2).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train(model, model_name, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dict)
        
        log[model_name] = test(model, test_loader, criterion, device)

    with open('fakeAV_trained_test_log.json', 'w') as json_file:
        json.dump(log, json_file, indent=4)