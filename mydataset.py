# library
# standard library

# third-party library
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
#import matplotlib.pyplot as plt

import io
#import requests
from PIL import Image
from torchvision import models, transforms

import torch.utils.data as data_utils
from resizeimage import resizeimage

HEIGHT = 32
defaultPath = '/home/xliu/crnn.pytorch/default.jpg'

normalize = transforms.Normalize(
   mean=[0.5],
   std=[0.5]
)
preprocess = transforms.Compose([   
   #transforms.Resize()
   transforms.ToTensor(),
   normalize
])

def default_loader(path):
    img = Image.open(path).convert('L')
    width, height = img.size
    m = preprocess(img.resize((int(width*HEIGHT/height),HEIGHT))) 
    #m = preprocess(resizeimage.resize_height(Image.open(path), 31).convert('L'))
    m.view(HEIGHT,-1)
    print(m.size())
    return m
    #return preprocess(resizeimage.resize_height(Image.open(path), 32).convert('L'))

with open('/home/xliu/crnn.pytorch/mnt/ramdisk/max/90kDICT32px/lexicon.txt', 'r') as lex:
    targets = []
    for line in lex:
        line = line.strip('\n')
        targets.append(line)
def default_target(index):
    return targets[index+1]

class myDataSet(data_utils.Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader        

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        #img = self.loader('mnt/ramdisk/max/90kDICT32px/'+fn)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        try:
            img = Image.open('mnt/ramdisk/max/90kDICT32px/'+fn).convert('L')
        except:
            img = Image.open(defaultPath).convert('L')
            print('This is a rubbish image:' + fn)
        return img,targets[label]

    def __len__(self):
        return len(self.imgs)
