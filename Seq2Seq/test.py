import pandas as pd
import numpy as np

# data = pd.read_csv()
#这个版本是主版本

class Config(object):
    def __init__(self):
        self.model_name = 'CNN-LSTM'

        self.inputSize = 255
        self.hiddenSize = 2




class Encoder():
    def __init__(self,config):
        super(Decoder, self).__init__()
        self.input_dim = config.inputSize
        self.hidden_dim = config.hiddenSize


    def forward(self):
        pass

class Decoder():
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
        pass

class Seq2Seq():
    def __init__(self,encoder,decoder):
        super(Seq2Seq, self).__init__()
