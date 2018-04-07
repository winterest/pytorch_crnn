import torch.nn as nn
from . import config

import torch
from torch.autograd import Variable

minW = config.minW 
minH = config.minH
scale = config.scale

nRoIFeature = sum(scale) 

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, nc, nclass, nh,  nRoIFeature=21, n_rnn=2,nRin=512, leakyRelu=False):# 1, 37
        super(CRNN, self).__init__()
        '''
        Revised version of original CNN-RNN model: CNN-RoI-FC-RNN
        
        CNN: b*1*H*W -- b*c*h*w -- (RoI) -- b*c*nFeature*n -- (FC) -- b*c*1*n -- (RNN) -- n*b*class 
        
        '''

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2,2), (1,1), (1,1)))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2,2), (1,1), (1,1)))  # 128x8x32
        convRelu(2, True)
        
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (1, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (1, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.fc = nn.Sequential(
            nn.Linear(nRoIFeature, 1),
            #nn.Linear(8, 1)
            )
            
        self.rnn = nn.Sequential(
            BidirectionalLSTM(nRin, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        print(input.size())
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        print(conv.size())

        h0 = int(max(minH, ((h+scale[3]-1)//scale[3])*scale[3]))
        #print('h0   :  '+str(h0))
        
        conv0 = Variable(torch.zeros(b,c,h0,w)).cuda()
        conv0[:,:,0:h,:] = conv
        #conv0 = conv0.view(b,c,h0,w)
        
        roi0 = nn.MaxPool2d((h0//scale[0],w//scale[0]))(conv0)
        roi1 = nn.MaxPool2d((h0//scale[1],w//scale[1]))(conv0)
        roi2 = nn.MaxPool2d((h0//scale[2],w//scale[2]))(conv0)
        roi3 = nn.MaxPool2d((h0//scale[3],w//scale[3]))(conv0)
        #print(roi2.size())
        roi0 = roi0.permute(0,1,3,2)
        roi1 = roi1.permute(0,1,3,2)
        roi2 = roi2.permute(0,1,3,2)
        roi3 = roi3.permute(0,1,3,2)
        
        roi = torch.cat((roi0,roi1,roi2,roi3),3)    #(b,c,n,nRoIFeature = 21)
        
        conv = self.fc(roi)    #(b,c,n,1)

        conv = conv.squeeze(3)

        conv = conv.permute(2, 0, 1)  # [n, b, c]     

        # rnn features
        output = self.rnn(conv)
        
        return output
