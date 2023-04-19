import os
import math
import pandas as pd
import numpy as np
import random as rn
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# 忽略tensorflow警告等级为2的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 在Keras开发过程中获取可复现的结果
os.environ['PYTHONHASHSEED'] = '0'

# 以下是Numpy在一个明确的初始状态生成固定随机数字
np.random.seed(42)

# 以下是Python在一个明确的初始状态生成固定随机数字所必需的。
rn.seed(12345)

# 强制 TensorFlow 使用单线程，多线程是结果不可复现的一个潜在因素
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# tf.set_random_seed() 将会以TensorFlow为后端，在一个明确的初始状态下生成固定随机数字
tf.compat.v1.set_random_seed(1234)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


def createDataSet(seq, timestep):
    """
    创建数据集
    :param seq: 输入
    :param timestep: 时间步
    :return: 数据集
    """
    X = []
    Y = []
    for i in range(len(seq)-timestep):
        X.append(np.array(seq)[:, np.newaxis][i:i + timestep])
        Y.append([np.array(seq)[i + timestep]])
    return np.array(X), np.array(Y)


def createInDataSet(seq, timestep):
    """
    创建数据集
    :param seq: 输入
    :param timestep: 时间步
    :return: 数据集
    """
    X = []
    for i in range(len(seq)-timestep):
        X.append(np.array(seq)[:, np.newaxis][i:i + timestep])
    return np.array(X)


def createOutDataSet(seq, timestep):
    """
    创建数据集
    :param seq: 输入
    :param timestep: 时间步
    :return: 数据集
    """
    Y = []
    for i in range(len(seq)-timestep):
        Y.append([np.array(seq)[i + timestep]])
    return np.array(Y)


def scale(raw_data):
    """
    最大最小归一化
    :param raw_data: 数据
    :return: 归一化后的数据
    """
    return (raw_data-np.min(raw_data))/(np.max(raw_data)-np.min(raw_data))


def scale_inv(inv_data, raw_data):
    """
    反归一化
    :param raw_data: 数据
    :return:
    """
    return inv_data*(np.max(raw_data)-np.min(raw_data))+np.min(raw_data)


def buildLSTM(timeStep, neuron):
    """
    搭建LSTM网络，激活函数为tanh
    :param timeStep: 时间步
    :return: LSTM模型
    """
    model = Sequential()  # 创建模型
    model.add(LSTM(neuron, input_shape=(timeStep, 1), return_sequences=True, activation='tanh'))  # 添加LSTM层
    model.add(Dropout(0.1))
    model.add(LSTM(neuron, activation='tanh'))  # 添加LSTM层
    model.add(Dropout(0.1))
    # model.add(LSTM(neuron, return_sequences=True, activation='tanh'))  # 添加LSTM层
    # model.add(Dropout(0.1))
    # model.add(LSTM(neuron, activation='tanh'))  # 添加LSTM层
    # model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.summary()
    return model


def train_model(In, Out, timestep, epochs, batchsize, neuron):
    """
    预测
    :param In: 模型输入
    :param Out: 模型输出
    :param timestep: 时间步
    :param epochs: 轮询次数
    :param batchsize: 批量大小
    :param neuron: 神经元数
    :return: RMSE,lstm
    """
    # LSTM模型
    lstm = buildLSTM(timeStep=timestep, neuron=neuron)

    # RMSE
    RMSE = list()

    # 十折交叉验证法
    KF = KFold(n_splits=10)
    for train_index, test_index in KF.split(In):
        scaleIn = scale(In)
        scaleOut = scale(Out)
        trainIn = scaleIn[train_index]
        trainOut = scaleOut[train_index]
        testIn = scaleIn[test_index]
        testOut = scaleOut[test_index]
        train_x = createInDataSet(trainIn, timestep)
        train_y = createOutDataSet(trainOut, timestep)
        test_x = createInDataSet(testIn, timestep)
        test_y = createOutDataSet(testOut, timestep)
        lstm.fit(train_x, train_y, epochs=epochs, verbose=0, batch_size=batchsize)  # 训练模型
        ypre = lstm.predict(test_x)  # 测试模型
        ypre = scale_inv(ypre, Out)  # 反归一化
        ytru = scale_inv(test_y, Out)  # 反归一化
        RMSE.append(math.sqrt(MSE(ytru, ypre)))

    return np.mean(RMSE), lstm


if __name__ == '__main__':

    # 设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

    # 读取数据
    Capacity = pd.read_csv('./LSTM模型时序数据/capacity_discharge_162.csv', header=None).iloc[:, 0].tolist()
    HI = pd.read_csv('./LSTM模型时序数据/HI_162.csv', header=None).iloc[:, 0].tolist()

    # # 批量大小
    # epoch = 200
    # neuron = 32
    # timeStep = 6
    # batch_size = [4, 8, 16, 32, 64]
    # pivot = list()
    # for batch in batch_size:
    #     pivot.append([batch])
    #     res = train_model(HI, Capacity, timeStep, epoch, batch, neuron)
    #     pivot.append([res[0]])
    #     with open('batch_size_162.csv', 'a', encoding='utf-8') as csvFile:
    #         np.savetxt(csvFile, np.array(pivot).T, delimiter=',', fmt='%s')
    #     pivot = list()
    #     print(batch, res[0])

    # # 神经元数
    # timeStep = 6
    # epoch = 200
    # bathSize = 8
    # neuronSet = [8, 16, 32, 64, 128]
    # pivot = list()
    # for neuron in neuronSet:
    #     pivot.append([neuron])
    #     res = train_model(HI, Capacity, timeStep, epoch, bathSize, neuron)
    #     pivot.append([res[0]])
    #     with open('neuron_162.csv', 'a', encoding='utf-8') as csvFile:
    #         np.savetxt(csvFile, np.array(pivot).T, delimiter=',', fmt='%s')
    #     pivot = list()
    #     print(neuron, res[0])

    # # epoch
    # timeStep = 6
    # bathSize = 8
    # neuron = 16
    # epochs = [32, 64, 128, 256]
    # pivot = list()
    # for epoch in epochs:
    #     pivot.append([epoch])
    #     # 当监测到loss停止改进时，结束训练。patience=2表示经过2个周期结果依旧没有改进，此时可以结束训练。
    #     early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    #     res = train_model(HI, Capacity, timeStep, epoch, bathSize, neuron)
    #     pivot.append([res[0]])
    #     with open('epoch_162.csv', 'a', encoding='utf-8') as csvFile:
    #         np.savetxt(csvFile, np.array(pivot).T, delimiter=',', fmt='%s')
    #     pivot = list()
    #     print(epoch, res[0])

    # # 隐藏层层数
    # bathSize = 8
    # neuron = 16
    # epoch = 128
    # timeStep = 6
    # # 当监测到loss停止改进时，结束训练。patience=2表示经过2个周期结果依旧没有改进，此时可以结束训练。
    # early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    # modelRMSE, lstmModel = train_model(HI, Capacity, timeStep, epoch, bathSize, neuron)
    # print(modelRMSE)
    # # 1层，0.029719163015731475
    # # 2层，0.0242811274155241
    # # 3层，0.03166134672717442
    # # 4层，0.03904292432451784

    # # timeStep
    # epoch = 128
    # bathSize = 8
    # neuron = 16
    # timeStep = [4, 6, 8, 10, 12]
    # pivot = list()
    # for step in timeStep:
    #     pivot.append([step])
    #     # 当监测到loss停止改进时，结束训练。patience=2表示经过2个周期结果依旧没有改进，此时可以结束训练。
    #     early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    #     modelRMSE, lstmModel = train_model(HI, Capacity, step, epoch, bathSize, neuron)
    #     pivot.append([modelRMSE])
    #     with open('timestep_162.csv', 'a', encoding='utf-8') as csvFile:
    #         np.savetxt(csvFile, np.array(pivot).T, delimiter=',', fmt='%s')
    #     pivot = list()
    #     print(step, modelRMSE)

    # 预测起点
    SPs = [50, 150]
    for SP in SPs:
        bathSize = 8
        neuron = 16
        epochs = 128
        timeStep = 4
        # 当监测到loss停止改进时，结束训练。patience=2表示经过2个周期结果依旧没有改进，此时可以结束训练。
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
        modelRMSE, lstmModel = train_model(HI, Capacity, timeStep, epochs, bathSize, neuron)
        print('modelRMSE:', modelRMSE)

        # 预测
        data = HI[SP-timeStep-1:]
        Tru = Capacity[SP-timeStep-1:]
        test = scale(data)
        test_x = createInDataSet(test, timeStep)
        pre = lstmModel.predict(test_x)
        pre = scale_inv(pre, Tru)
        tru = Capacity[SP-1:]
        # 存储预测结果
        with open('./LSTM预测原始数据/lstm_pre_162_%s.csv' % SP, 'a', encoding='utf-8') as csvFile:
            np.savetxt(csvFile, pre, delimiter=',', fmt='%s')

        # 存储RMSE、MAE、R2结果
        res = list()
        print('----------RMSE-----------')
        print(math.sqrt(MSE(tru, pre)))
        print('----------MAE-----------')
        print(MAE(tru, pre))
        print('----------R2-----------')
        print(R2(tru, pre))
        res.append(math.sqrt(MSE(tru, pre)))
        res.append(MAE(tru, pre))
        res.append(R2(tru, pre))
        with open('./LSTM预测原始数据/lstm_rmse_mae_r2_162_%s.csv' % SP, 'a', encoding='utf-8') as csvFile:
            np.savetxt(csvFile, np.array(res), delimiter=',', fmt='%s')

    # plt.plot(tru, label='真实')
    # plt.plot(pre, label='预测')
    # plt.legend()
    # plt.show()
