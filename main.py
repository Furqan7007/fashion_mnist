import glob
import os
import numpy as np
import pandas as pd
import argparse
from loader import *
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser("Training on Fashion MNIST")
parser.add_argument("-d","--data_path", type = str, help="Data path for fashion mnist")
parser.add_argument("-c","--csv_path", type=str, help="CSV path")
parser.add_argument("-b","--batch_size", type = int, default = 128)
parser.add_argument("-e","--epochs", type=int, default=50)

args = parser.parse_args()

def train_val_split(csv_file):
    df = pd.read_csv(os.path.join(args.csv_path))
    # print(len(df))
    X = df.iloc[:,0]
    Y = df.iloc[:,1]
    X_train, y_train, X_val, y_val = train_test_split(X,Y, train_size=0.8)
    return X_train, y_train, X_val, y_val

def train(model, data_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for idx, sample in enumerate(data_loader):
        data = sample["image"]
        label = sample["label"]
    output = model(data)
    loss = criterion(output, label)


def val(model, val_loader, epoch):
    model.eval()
    


def main():
    
    batch_size = args.batch_size
    epochs = args.epochs

    mnist_dataset = FashionMNISTDataset(args.data_path, args.csv_path)
    train_dataset , val_dataset = torch.utils.data.random_split(mnist_dataset, [50000,10000])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size,
                         num_workers = 2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=2,
                            shuffle=True)
    
    optimizer = optim.sgd(lr=0.01)
    
    max_acc = 0
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch)

        accuracy = val(model, val_loader, epoch)
        if accuracy > max_acc:
            max_acc = accuracy
        



    

if __name__=="__main__":
    main()
