import os
import argparse

from model.utils import *  #每个函数内部的方法
from model.DA_RNN import *
from importlib import import_module  #动态加载不同的模块
import train
import utils   #这个为计算时间的方法，为公共方法，所以定义在外面
from model.Random_Forest import *
from model.ARIMA import *
from model.GBRT import *

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="Seq2Seq类模型进行时间序列预测")

    #选择模型即可
    parser.add_argument('--model_name',type=str,default='DA_test2',help='choose a model CNN,LSTM,GRU,Seq2Seq,CNN_LSTM+self_Att,Seq2Seq+Att,DA_RNN,CNN_LSTM,CNN_GRU')

    args = parser.parse_args()

    return args

def main_DA_RNN():
    """
    该函数为训练Seq2Seq_Att模型的函数
    :return:
    """
    args = parse_args() #加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.'+ model_name)
    config = x.Config()

    # Read dataset
    print("==> Load dataset ...")
    X, y = read_data(config.dataroot, debug=False)

    # Initialize model
    print("==> Initialize DA_RNN model ...")
    model = x.Model(
        X,
        y,
        config
    )

    # Train
    print("==> Start training ...")
    model.train(model,config)  # model 输入进去用于保存模型

    # Prediction
    y_pred = model.test()

    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.savefig("./data/pic/1.png")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.savefig("./data/pic/2.png")
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(y_pred, label='Predicted')
    plt.plot(model.y[model.train_timesteps:], label="True")
    plt.legend(loc='upper left')
    plt.savefig("./data/pic/3.png")
    plt.close(fig3)
    print('Finished Training')

def main_DA_test():
    """
        该函数为训练Seq2Seq_Att模型的函数
        :return:
        """
    args = parse_args()  # 加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.' + model_name)
    config = x.Config()

    # Read dataset
    print("==> Load dataset ...")
    X, y = read_data(config.dataroot, debug=False)

    # Initialize model
    print("==> Initialize DA_test model ...")
    model = x.Model(
        X,
        y,
        config
    )

    # Train
    print("==> Start training ...")
    model.train(model, config)  # model 输入进去用于保存模型

    # Prediction
    y_pred = model.test()

    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.savefig("./data/pic/1_test.png")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.savefig("./data/pic/2_test.png")
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(y_pred, label='Predicted')
    plt.plot(model.y[model.train_timesteps:], label="True")
    plt.legend(loc='upper left')
    plt.savefig("./data/pic/3_test.png")
    plt.close(fig3)
    print('Finished Training')


def run_DA_RNN_model():
    """

    :return:
    """
    print("*"*50)

    args = parse_args()  # 加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.' + model_name)
    config = x.Config()

    print('==>加载数据...')
    X, y = read_data(config.dataroot, debug=False)
    print('==>运行已经跑好的模型:',model_name)
    model = x.Model(
        X,
        y,
        config
    )

    model.load_state_dict(torch.load(config.save_model))
    print('==>模型加载成功！')
    model.test_model(config)

def main_Seq2Seq():
    """
    模型为Seq2Seq 模型的代码
    :return:
    """
    args = parse_args()  # 加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.' + model_name)
    config = x.Config()  # 加载模型的配置
    model = x.Seq2Seq(config)  # 加载模型
    print('==>当前使用的模型为：' + model_name)

    print('==>加载数据中...')
    # ntime_steps   为时间窗口T
    # n_next        为想要预测的天数

    Train_X, Train_Y, Test_X, Test_Y = get_data(config.ntime_steps, config.n_next)

    train_data = MyDataset(Train_X, Train_Y)
    train_dataloader = DataLoader(dataset=train_data, batch_size=config.batch_size)


    model.train(model, config, train_dataloader)  # Seq2Seq训练模型

    # model.test_model()  # 测试还有问题

def run_Seq2Seq_model():
    print("*"*100)

    args = parse_args()  # 加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.' + model_name)
    config = x.Config()

    print('运行已经跑好的模型')
    model = x.Seq2Seq(config)
    model.load_state_dict(torch.load(config.save_model))
    model.test_model()    # 测试有问题

#随机森林方法预测
def RF():
    # ntime_steps = 10
    # n_next = 1
    # Train_X, Train_Y, Test_X, Test_Y = get_data(ntime_steps, n_next, hasTest=True)

    X, y = read_data('./data/one_hot_甘.csv', debug=False)

    train_size = int(len(X) * 0.8)

    train_x = X[:train_size]
    test_x = X[train_size:]

    train_y = y[:train_size]
    test_y = y[train_size:]

    trainRF(train_x,train_y,test_x,test_y,y)

#移动平均算法
def ARIMA():
    X, y = read_data('./data/one_hot_甘.csv', debug=False)

    train_size = int(len(X) * 0.8)

    train_x = X[:train_size]
    test_x = X[train_size:]

    train_y = y[:train_size]
    test_y = y[train_size:]

    train_ARIMA(train_x, train_y, test_x, test_y, y)

    draw_arima()


def GBRT():
    X, y = read_data('./data/one_hot_甘.csv', debug=False)

    train_size = int(len(X) * 0.8)

    train_x = X[:train_size]
    test_x = X[train_size:]

    train_y = y[:train_size]
    test_y = y[train_size:]

    train_GBRT(train_x, train_y, test_x, test_y, y)


if __name__ == '__main__':

    # RF()  #随机森林
    # ARIMA()  #移动平均算法

    # GBRT()   #渐进递归回归树（Gradient Boosting Regression）


    # main_DA_RNN()
    # main_DA_test()
    run_DA_RNN_model()
    # main_Seq2Seq()
    #
    # run_Seq2Seq_model()  #有问题

#############################################################
    #   CNN_LSTM   CNN_GRU

    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样
    # #
    # args = parse_args()  # 加载所选模型的名字
    # model_name = args.model_name
    # x = import_module('model.' + model_name)
    # config = x.Config()
    #
    # model = x.Model(config)
    # model = model.to(config.device)
    #
    # print('==>当前使用的模型为：' + model_name +'_T：'+str(config.ntime_steps) + '_D:'+ str(config.n_next))
    #
    # print('==>加载数据中...')
    #
    # Train_X, Train_Y, Test_X, Test_Y = get_data(config.ntime_steps, config.n_next)
    #
    # # ntime_steps   为时间窗口T
    # # n_next        为想要预测的天数
    # train_data = MyDataset(Train_X,Train_Y)
    #
    # test_data = MyDataset(Test_X,Test_Y)
    #
    #
    # train_dataloader = DataLoader(dataset=train_data, batch_size=config.batch_size)
    #
    # test_dataloader = DataLoader(dataset=test_data,batch_size=config.test_batch_size)
    #
    # flag = os.path.exists(config.save_model)
    # if flag==True:
    #     print('====>该模型已经训练过！直接进行测试')
    #     train.test(model, config)
    #     # train.draw_model_structure(model, config)  # 画出模型的结构
    #
    # else:
    #     train.train(model, config, train_dataloader,test_dataloader)
    #
    #     train.test(model,config)
    # #
    # #     # train.draw_model_structure(model,config)  #画出模型的结构


















