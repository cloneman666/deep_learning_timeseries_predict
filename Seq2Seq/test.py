import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Config(object):
    def __init__(self):
        self.model_name = 'CNN-LSTM'


        #加密部分参数encoder
        self.encoder_inputSize = 16
        self.encoder_hiddenSize = 64
        self.dropout = 0.1
        self.encoder_numlayers = 1

        self.encoder_outputSize = 10

        self.batch_first = True
        self.bidirectional = True

        #解密部分参数
        self.decoder_hiddenSize = 64
        self.decoder_numlayers = 1
        self.decoder_outputSize = 10  #此项为最终分类的数目

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lr = 0.0005

class Encoder(nn.Module):
    def __init__(self,config):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(input_size=config.encoder_inputSize,hidden_size=config.encoder_hiddenSize,
                           num_layers=config.encoder_numlayers,
                           dropout=(0 if config.encoder_numlayers == 1 else config.dropout),
                           batch_first=config.batch_first,bidirectional=config.bidirectional
                           )
        if config.bidirectional: #是否为双向结构
            self.fc = nn.Linear(config.hiddenSize * 2,config.encoder_outputSize)

        self.fc = nn.Linear(config.hiddenSize,config.encoder_outputSize)


    def forward(self,x): #数据的样子要根据具体情况进行调整
        output, (h_n, c_n) = self.rnn(x)
        result = self.fc(output)
        return result



class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(input_size=config.encoder_outputSize,hidden_size=config.decoder_hiddenSize,
                           num_layers=config.decoder_numlayers,dropout=(0 if config.decoder_numlayers ==1 else config.dropout),
                           batch_first=config.batch_first,bidirectional=config.bidirectional
                           )
        if config.bidirectional:
            self.fc = nn.Linear(config.decoder_hiddenSize * 2,config.decoder_outputSize)

        self.fc = nn.Linear(config.decoder_hiddenSize,config.decoder_outputSize)

    def forward(self,x): #数据的样子要根据具体情况进行调整
        output, (h_n, c_n) = self.rnn(x)
        result = self.fc(output)
        return result

class Seq2Seq(nn.Module):
    def __init__(self,):
        super(Seq2Seq, self).__init__()
        self.config = Config()
        self.encoder = Encoder(self.config).to(self.config.device)
        self.decoder = Decoder(self.config).to(self.config.device)

        #损失函数,优化函数等定义
        self.criterion = nn.CrossEntropyLoss().to(self.config.device)

        self.encoder_optimer = optim.Adam(params=filter(lambda p: p.requires_grad,self.encoder.parameters()),
                                            lr = self.config.lr)

        self.decoder_optimer = optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                          lr=self.config.lr)

    def forward(self,x):
        pass



