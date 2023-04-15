import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.model import FaceRecognizer
from utils.training import Trainer, TrainRecord

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import math

# HYPER PARAMS
NUM_EPOCHS = 100
BATCH_SIZE = 128
lrate = 0.0009

stop_avg_loss = 1e-8

# DATA COLLECT

DATA_FOLDER = "./data"
MODEL_STATE_DICT_PATH = "./model/model.pt"
MODEL_CLASS2ID = "./model/class2id.csv"

class BKStudentFaceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).to(torch.float64) / 255,
        self.y = torch.from_numpy(y).to(torch.float64),
        self.num_points = num_points

    def __getitem__(self, index):
        return self.x[0][index], self.y[0][index]

    def __len__(self):
        return self.num_points

x, y = [], []
class2id, id2class = {}, {}

num_points = 0
num_classes = 0

for img_file in os.listdir(DATA_FOLDER):
    data_path = os.path.join(DATA_FOLDER, img_file)
    img = cv2.imread(data_path, 0)
    label, _ = img_file.split("_")


    if label not in id2class.values():
        id2class[num_classes] = label
        class2id[label] = num_classes
        num_classes += 1
     
    x.append(img)
    y.append(class2id[label])
    num_points += 1

x = np.array(x)
y = np.array(y, dtype = np.int64)
y = torch.from_numpy(y)
y = np.array(nn.functional.one_hot(y, num_classes = num_classes))

train_dataset = BKStudentFaceDataset(x, y)
train_dataloader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

# MODEL
model = FaceRecognizer(num_classes)

# LOSS AND OPTIMIZER
loss_fn = nn.CrossEntropyLoss(reduction = "mean")
optimizer = torch.optim.Adam(
    params = model.parameters(),
    lr = lrate
)

# TRAINING
guard = Trainer()
record = TrainRecord()

num_batches = math.ceil(train_dataset.num_points / BATCH_SIZE)

stop_flag = False

print("Training started ...")

for epoch in range(NUM_EPOCHS):
    for batch, (features, labels) in enumerate(train_dataloader):
        labels_predict = model(features.reshape(features.shape[0], 1, features.shape[1], features.shape[2]))
        loss = loss_fn(labels_predict, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        record.record_state_dict(model.state_dict())

        record.record(
            curr_epoch = epoch,
            curr_batch = batch,
            loss = loss.detach().item()
        )

        if epoch > 0:
            # print(record.pre_avg_loss, record.curr_avg_loss, guard.patience)
            if guard.check(record.best_avg_loss, record.curr_avg_loss):
                stop_flag = True
                print("Stopped by Trainer! He is angry now ...")
                break
        
        if record.best_avg_loss < stop_avg_loss:
            print("\nFinal loss: {}".format(record.curr_loss))
            stop_flag = True
            break
        
        record.update_best_avg_loss()
        
        if batch % 10 == 0:
            record.report(NUM_EPOCHS, num_batches)
            guard.comment()

        if (epoch + 1) % 20 == 0 and batch == 0:
            record.add_subplot()

    if stop_flag:
        break

print("Finish training")
record.plot()

save = input("Save model? ")
if save == "OK":
    print("Model saved!")
    class2id = pd.DataFrame([class2id])
    class2id.to_csv(MODEL_CLASS2ID, index = False)

    torch.save(record.state_dict, MODEL_STATE_DICT_PATH)
