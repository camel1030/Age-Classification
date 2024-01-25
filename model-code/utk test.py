# Imports

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image, decode_image
from PIL import Image
from torchvision import transforms
import PIL
import cv2
import resnet
import torch.optim as optim
from tqdm import tqdm
import wandb

class CustomImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file_name'].values
        self.img_labels = df['age'].values
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        # image = np.array(read_image(img_path))
        label = self.img_labels[idx]
        label = np.asarray(label, dtype=np.uint8)
        label = torch.from_numpy(label)
        if self.transform:
            image = self.transform(image)
        return image, label


cuda = 0
seed = 1
IMP_WEIGHT = 1
 # Hyperparameters
learning_rate = 0.0005
num_epochs = 2

# Architecture
NUM_CLASSES = 40
BATCH_SIZE = 265
GRAYSCALE = False

TRAIN_CSV_PATH = 'C:/Users/jeong/Desktop/code/utk/part4/utkimagetrain.csv'
TEST_CSV_PATH = 'C:/Users/jeong/Desktop/code/utk/part4/utkimageval.csv'
Train_IMAGE_PATH = 'C:/Users/jeong/Desktop/code/utk/train/'
Val_IMAGE_PATH = 'C:/Users/jeong/Desktop/code/utk/val/'

trans = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])  #이미지 크기 고정
target_transform = transforms.Compose([transforms.ToTensor()])
training_data = CustomImageDataset(csv_path=TRAIN_CSV_PATH, img_dir=Train_IMAGE_PATH, transform=trans)
test_data = CustomImageDataset(csv_path=TEST_CSV_PATH, img_dir=Val_IMAGE_PATH, transform=trans)

outpath = 'utkmodel'
if cuda >= 0:
    DEVICE = torch.device("cuda:%d" % cuda)
else:
    DEVICE = torch.device("cpu")

if seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = seed

PATH = outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)
LOGFILE = os.path.join(PATH, 'training.log')

# Logging

header = []

header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA DEVICE available: %s' % torch.cuda.is_available())
header.append('Using CUDA DEVICE: %s' % DEVICE)
header.append('Random Seed: %s' % RANDOM_SEED)
header.append('Task Importance Weight: %s' % IMP_WEIGHT)
header.append('Output Path: %s' % PATH)

with open(LOGFILE, 'w') as f:
    for entry in header:
        print(entry)
        f.write('%s\n' % entry)
        f.flush()

##########################
# SETTINGS
##########################
df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
ages = df['age'].values
del df
ages = torch.tensor(ages, dtype=torch.float)
train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=100, shuffle=False)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
model = resnet.resnet34(NUM_CLASSES, GRAYSCALE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss().to(DEVICE)
config = {
        'model_name' : 'ResNet34',
        'batch_size' : train_dataloader.batch_size,
        'epoch' : num_epochs,
        'criterion' : loss_fn,
        'optimizer' : optimizer,
    }
### 새 wandb 세션 시작 ###
wandb.init(reinit=True, project='test', config=config)
### 모델 파라미터(그래디언트 등) 추적을 위한 .watch 호출 ###
wandb.watch(model)
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_dataloader)
    for batch_idx, (features, targets) in enumerate(loop):
        loop.set_description(f"Epoch {epoch + 1}")
        features = features.to(DEVICE)
        targets = targets.to(DEVICE).unsqueeze(1).to(torch.float32)
        y = model(features)
        #비용함수 넣어야됨
        loss = loss_fn(y.to(torch.float32), targets.to(torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # LOGGING
        #
        # if batch_idx % 10:
        #     print(len(training_data)//BATCH_SIZE+1)
        #     s = ("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" % (epoch+1, num_epochs, batch_idx, len(training_data)//BATCH_SIZE+1, loss))
        #     print(s)
        #     with open(LOGFILE, 'a') as f:
        #         f.write('%s\n' % s)
        loop.set_postfix(loss=loss.item())
        wandb.log({'train_loss': loss})
    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference
        for batch_idx, (features, targets) in enumerate(test_dataloader):
            inputs, target = features.to(DEVICE), targets.to(DEVICE).unsqueeze(1).to(torch.float32)
            outputs = model(inputs)
            y = torch.max(outputs, 1)  # values, indics
            loss = loss_fn(y, targets.to(torch.float32))
            print(loss)
torch.save(model.state_dict(), os.path.join(PATH, 'model.pt'))
wandb.finish()

    #     total += target.size(0)
    #     correct += (predicted == target).sum().item()
    # print("test accuracy : {}%".format((100 * correct / total)))
#
#     print(train_mae)
#     test_mae, test_mse = compute_mae_and_mse(model, test_loader, DEVICE=DEVICE)
#
#     s = 'MSE: | Train: %.2f/%.2f | Test: %.2f/%.2f' % (train_mae, torch.sqrt(train_mse), test_mae, torch.sqrt(test_mse))
#     print(s)
#     with open(LOGFILE, 'a') as f:
#         f.write('%s\n' % s)
#
# s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
# print(s)
# with open(LOGFILE, 'a') as f:
#     f.write('%s\n' % s)
#
#     m = model.to(torch.DEVICE('cpu'))
#     torch.save(model.state_dict(), os.path.join(PATH, 'model.pt'))