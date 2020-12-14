#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 12/7/20 9:38 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : draw.py
# @Software: PyCharm

import matplotlib.pyplot as plt
x=[3,7,11,15,20]



#  MSE
MSEARIMA = [0.42475,0.30813,0.28775,0.36860,0.34956]

MSECNN = [0.11316,0.10948,0.12121,0.14256,0.14521]
MSECNN_GRU = [0.06043,0.08405,0.20494,0.30298,0.35945]
MSECNN_LSTM = [0.04544,0.07019,0.05340,0.07802,0.08780]
MSECNN_LSTM_Att = [0.05284,0.07812,0.07220,0.07285,0.08931]
MSELSTM = [0.03848,0.14405,0.16397,0.21345,0.26437]
MSELSTM_CNN = [0.09895,0.08562,0.09928,0.09836,0.10385]
MSELSTM_Att = [0.13801,0.10653,0.33598,0.21508,0.38538]
MSEGRU = [0.04516,0.09458,0.16335,0.23761,0.27014]
MSEGRU_Att = [0.05315,0.20116,0.20184,0.26873,0.25894]
MSESeq2Seq = [0.00001,0.00005,0.00007,0.00020,0.00033]
MSESeq2Seq_Att = [0.00001,0.00008,0.00002,0.00006,0.00015]

plt.title('T:30')

# plt.plot(x,MSECNN,'*-',label='CNN')
# plt.plot(x,MSEGRU,label='GRU')
# plt.plot(x,MSEGRU_Att,label='GRU_Att')

# plt.plot(x,MSELSTM,label='LSTM')
# plt.plot(x,MSELSTM_Att,label='LSTM_Att')
# plt.plot(x,MSECNN_GRU,label='CNN_GRU')
plt.plot(x,MSEARIMA,'*-',label='ARIMA')
plt.plot(x,MSECNN_LSTM,'o-',label='CNN_LSTM')
plt.plot(x,MSECNN_LSTM_Att,'v-',label='CNN_LSTM_Att')
plt.plot(x,MSELSTM_CNN,'+-',label='LSTM_CNN')
plt.plot(x,MSESeq2Seq,'<-',label='Seq2Seq')
plt.plot(x,MSESeq2Seq_Att,'>-',label='Seq2Seq_Att')


plt.ylabel('MSE')
plt.xlabel('Future time step')
plt.legend()
plt.tight_layout()
plt.savefig('MSE各个算法对比.png',dpi=300)
plt.show()

###  RMSE
plt.title('T:30')
RMSEARIMA = [0.65173,0.55509,0.53642,0.60712,0.59123]

RMSECNN= [0.33639,0.33088,0.34816,0.37758,0.38106]
RMSECNN_LSTM = [0.21317,0.26493,0.23109,0.27932,0.29630]
RMSECNN_LSTM_Att = [0.22987,0.27950,0.26870,0.26991,0.29885]
RMSELSTM_CNN = [0.31456,0.29261,0.31509,0.31363,0.32226]
RMSESeq2Seq = [0.00249,0.00720,0.00832,0.01398,0.01816]
RMSESeq2Seq_Att = [0.00237,0.00890,0.00488,0.00773,0.01205]

# plt.plot(x,RMSECNN,'*-',label='CNN')
plt.plot(x,RMSEARIMA,'*-',label='ARIMA')
plt.plot(x,RMSECNN_LSTM,'o-',label='CNN_LSTM')
plt.plot(x,RMSECNN_LSTM_Att,'v-',label='CNN_LSTM_Att')
plt.plot(x,RMSELSTM_CNN,'+-',label='LSTM_CNN')
plt.plot(x,RMSESeq2Seq,'<-',label='Seq2Seq')
plt.plot(x,RMSESeq2Seq_Att,'>-',label='Seq2Seq_Att')

plt.ylabel('RMSE')
plt.xlabel('Future time step')
plt.legend()
plt.tight_layout()
plt.savefig('RMSE各个算法对比.png',dpi=300)
plt.show()

### MAE
plt.title('T:30')

MAEARIMA = [0.56223,0.50075,0.46643,0.54624,0.53631]
MAECNN =[0.18713,0.18357,0.19309,0.20970,0.21292]
MAECNN_LSTM = [0.10415,0.13326,0.11096,0.13180,0.15966]
MAECNN_LSTM_Att = [0.12232,0.12005,0.12633,0.13253,0.13789]
MAELSTM_CNN = [0.19656,0.16951,0.19683,0.19623,0.19938]
MAESeq2Seq = [0.00168,0.00490,0.00621,0.01084,0.01408]
MAESeq2Seq_Att = [0.00177,0.00694,0.00380,0.00605,0.00967]

# plt.plot(x,MAECNN,'*-',label='CNN')
plt.plot(x,MAEARIMA,'*-',label='ARIMA')
plt.plot(x,MAECNN_LSTM,'o-',label='CNN_LSTM')
plt.plot(x,MAECNN_LSTM_Att,'v-',label='CNN_LSTM_Att')
plt.plot(x,MAELSTM_CNN,'+-',label='LSTM_CNN')
plt.plot(x,MAESeq2Seq,'<-',label='Seq2Seq')
plt.plot(x,MAESeq2Seq_Att,'>-',label='Seq2Seq_Att')

plt.ylabel('MAE')
plt.xlabel('Future time step')
plt.legend()
plt.tight_layout()
plt.savefig('MAE各个算法对比.png',dpi=300)
plt.show()