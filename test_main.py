
import os, importlib
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.LAV_small import LAVDF_small
from dataset.FakeAVCeleb import FakeAVCeleb
from train import *
from model import *

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = 'C:/Users/User/Downloads/FakeAVCeleb_v1.2'
    #root_dir = 'C:/Users/User/Downloads/LAV-small'


    train_dataset = FakeAVCeleb(root_dir=root_dir, split='train', num_samples=5000)
    val_dataset = FakeAVCeleb(root_dir=root_dir, split='val', num_samples=200)
    test_dataset = FakeAVCeleb(root_dir=root_dir, split='test', num_samples=1000)

    num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=test_dataset.collate_fn)

    for video, audio, label in test_loader:
        print(f"Train Video shape: {video.shape}, {audio.shape}, Label: {label}")
        break

    models = [
                "CNN_LSTM_each_AV", 
                "CNN_LSTM_cross_attention_AV", 
                #"CNN_BiMAMBA_each_AV", 
                #"CNN_BiMAMBA_MHA_AV",
                #"CNN_Transformer_AV",
                #"MeseInception_AV",
                #"Resnet_LSTM_AV",
                "KAN_CNN_LSTM_each_AV",
                #"KAN_CNN_Transformer_AV",
                "AVT_DWF_AV",
                "CNN_LSTM_V",
                "AVT_MAMBA_AV",
                "CNN_BiMAMBA_AV"
            ]
    test_log = dict()

    save_dict = "./trained_FakeAVCeleb"
    criterion = nn.CrossEntropyLoss()

    for model_name in models:
        print(model_name)
        module = importlib.import_module(f"model.{model_name}")
        model_class = getattr(module, model_name)
        model = model_class(num_classes=2).to(device)
        model.load_state_dict(torch.load(f'{save_dict}/{model_name}.pth'))
        
        acc = test(model, test_loader, criterion, device)
        test_log[model_name] = acc
    print(test_log)

    with open('fakeAV_trained_test_log.json', 'w') as json_file:
        json.dump(test_log, json_file, indent=4)