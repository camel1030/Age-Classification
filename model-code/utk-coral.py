# coding: utf-8

#############################################
# Consistent Cumulative Logits with ResNet-34
#############################################

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

def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    #print(uniq)       #shape: 104
    num_examples = label_array.size(0)  #22900
    m = torch.zeros(uniq.shape[0])
    #print(torch.max(uniq))       #104
    for i, t in enumerate(uniq):  #나이 최소~최대 순서로
        #print(t)# tensor 115
        #print(label_array[label_array > t].size(0))
        #print(num_examples - label_array[label_array > t].size(0))
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0), 
                                      num_examples - label_array[label_array > t].size(0)]))    #115
        m[i] = torch.sqrt(m_k.float())  #m은 104인데 m_k는 115

    imp = m/torch.max(m)
    return imp

###################
# Dataset
###################


class UTKFaceDataset(Dataset):
    """Custom Dataset for loading UTKFace face images"""

    def __init__(self,
                 csv_path, img_dir, transform):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file_name'].values
        self.y = df['age'].values
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        img = Image.open(img_path).convert("RGB")
        img = transforms.Resize((256,256))(img)
        img = self.transform(img)
        label = self.y[index]
        levels = [1]*label + [0]*(NUM_CLASSES - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)

        return img, label, levels

    def __len__(self):
        return self.y.shape[0]

##########################
# MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas

def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model

###########################################
# Initialize Cost, Model, and Optimizer
###########################################

def cost_fn(logits, levels, imp):
    #print(logits.shape)
    #print(levels.shape)
    a = F.logsigmoid(logits)*levels
    val = (-torch.sum((a + (F.logsigmoid(logits) - logits)*(1-levels))*imp, dim=1))
    return torch.mean(val)

def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    print(data_loader.shape)
    for i, (features, targets, levels) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets)**2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return features, targets

TRAIN_CSV_PATH = 'C:/Users/jeong/Desktop/code/utk/part4/utkimagetrain.csv'
TEST_CSV_PATH = 'C:/Users/jeong/Desktop/code/utk/part4/utkimagetest.csv'
IMAGE_PATH = 'C:/Users/jeong/Desktop/code/utk/part4/'

cuda=0
seed=1
IMP_WEIGHT=1
 # Hyperparameters
learning_rate = 0.0005
num_epochs = 1

# Architecture
NUM_CLASSES = 40
BATCH_SIZE = 256
GRAYSCALE = False
class PrintTransform:
    def __call__(self, x):
        if isinstance(x, PIL.Image.Image):
            print(x.size)
        elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
            print(x.shape)
        return x

def run():
    

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
    header.append('CUDA device available: %s' % torch.cuda.is_available())
    header.append('Using CUDA device: %s' % DEVICE)
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
    # Data-specific scheme
    if IMP_WEIGHT == 0:
        imp = torch.ones(NUM_CLASSES-1, dtype=torch.float)
    elif IMP_WEIGHT == 1:
        imp = task_importance_weights(ages)
        imp = imp[0:NUM_CLASSES-1]
    else:
        raise ValueError('Incorrect importance weight parameter.')
    imp = imp.to(DEVICE)


    custom_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomCrop((120, 120))]) #PrintTransform()

    train_dataset = UTKFaceDataset(csv_path=TRAIN_CSV_PATH,
                                    img_dir=IMAGE_PATH,
                                    transform=custom_transform)
    
    test_dataset = UTKFaceDataset(csv_path=TEST_CSV_PATH,
                                    img_dir=IMAGE_PATH,
                                    transform=custom_transform)
    from torch.nn.utils.rnn import pad_sequence
    
    def my_collate(batch):      #tensor 사이즈 맞추기 위해서 추가 but error....
        Image = torch.stack([item[0] for item in batch])
        label = [item[1] for item in batch]
        label = torch.LongTensor(label)
        bi = []
        min_len = min([len(item[2]) for item in batch])
        #print(max_len)
        for _x in [item[2] for item in batch]:
            tensor_len = _x.size(dim=0) # 텐서 길이
            p2d = (0, min_len - tensor_len)
            # right 에만 max_len - tensor_len 만큼 0으로 채워줄 것이다.
            _x = F.pad(_x, p2d)
            bi.append(_x)
        bi = torch.stack(bi)
        #print(bi.shape)
        return Image, label, bi
    
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=0,collate_fn=my_collate)
                                
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=1)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    model = resnet34(NUM_CLASSES, GRAYSCALE)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets, levels) in enumerate(train_loader):
            features = features.to(DEVICE)  #image
            #print(features)
            targets = targets
            targets = targets.to(DEVICE)    #나이
            levels = levels.to(DEVICE)      #binary 90
            #print(levels) #bi
            # FORWARD AND BACK PROP
            logits, probas = model(features)    #39
            #print(logits.shape)
            cost = cost_fn(logits, levels, imp)
            optimizer.zero_grad()

            cost.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()
            print(batch_idx%50)
            # LOGGING
            if not batch_idx % 10:
                s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'% (epoch+1, num_epochs, batch_idx,len(train_dataset)//BATCH_SIZE, cost))
                print(s)
                with open(LOGFILE, 'a') as f:
                    f.write('%s\n' % s)
        

        s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)
    print('Learning finished')
    
    
    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference

        train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                               device=DEVICE)
        print(train_mae)
        test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                             device=DEVICE)

        s = 'MAE/RMSE: | Train: %.2f/%.2f | Test: %.2f/%.2f' % (train_mae, torch.sqrt(train_mse), test_mae, torch.sqrt(test_mse))
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

    model = model.to(torch.device('cpu'))
    torch.save(model.state_dict(), os.path.join(PATH, 'model.pt'))

if __name__ == '__main__':
    run()