import os
import scipy.io as scio
import numpy as np


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files  # 当前路径下所有非目录子文件


def get_data(dataFile, fileName):
    battery = scio.loadmat(dataFile)
    cycle = battery[fileName]
    data = cycle[0, 0]['cycle']
    nums = data.shape[1]
    cut_off_voltage_discharge = [2.7, 2.5, 2.2, 2.5]  # 放电截止电压

    for i in range(nums):
        Type = data[0, i]['type']
        if Type == 'discharge':
            # 放电电压
            voltage_discharge = data[0, i]['data'][0, 0]['Voltage_measured'][0].tolist()
            # 放电时间
            time_discharge = data[0, i]['data'][0, 0]['Time'][0].tolist()
            # 放电容量
            capacity_discharge = data[0, i]['data'][0, 0]['Capacity']

            # 确定放电至放电截止电压时的数据长度
            count = 1
            for j in range(len(voltage_discharge)):
                if voltage_discharge[j] < cut_off_voltage_discharge[int(fileName[-1])-5]:
                    break
                else:
                    count += 1

            # 新放电电压
            voltage_discharge = voltage_discharge[:count]
            voltage_discharge = np.array(voltage_discharge)[:, np.newaxis]

            # 新放电时间
            time_discharge = time_discharge[:count]
            time_discharge = np.array(time_discharge)[:, np.newaxis]

            # 放电电压
            with open('./DataSet/%s/voltage_discharge.csv' % fileName, 'a', encoding='utf-8') as csvFile:
                np.savetxt(csvFile, voltage_discharge.T, delimiter=',', fmt='%.4f')

            # 放电时间
            with open('./DataSet/%s/time_discharge.csv' % fileName, 'a', encoding='utf-8') as csvFile:
                np.savetxt(csvFile, time_discharge.T, delimiter=',', fmt='%.4f')

            # 放电容量
            with open('./DataSet/%s/capacity_discharge.csv' % fileName, 'a', encoding='utf-8') as csvFile:
                np.savetxt(csvFile, capacity_discharge, delimiter=',', fmt='%.4f')

            # 健康因子
            AVF = [np.array(np.mean(voltage_discharge))]
            with open('./DataSet/%s/HI_%s.csv' % (fileName, fileName), 'a', encoding='utf-8') as csvFile:
                np.savetxt(csvFile, AVF, delimiter=',', fmt='%.4f')

        if Type == 'charge':
            # 充电电压
            voltage_charge = data[0, i]['data'][0, 0]['Voltage_measured']
            # 充电电流
            current_charge = data[0, i]['data'][0, 0]['Current_measured']
            # 充电时间
            time_charge = data[0, i]['data'][0, 0]['Time']

            # 充电电压
            with open('./DataSet/%s/voltage_charge.csv' % fileName, 'a', encoding='utf-8') as csvFile:
                np.savetxt(csvFile, voltage_charge, delimiter=',', fmt='%.4f')

            # 充电时间
            with open('./DataSet/%s/time_charge.csv' % fileName, 'a', encoding='utf-8') as csvFile:
                np.savetxt(csvFile, time_charge, delimiter=',', fmt='%.4f')

            # 充电电流
            with open('./DataSet/%s/current_charge.csv' % fileName, 'a', encoding='utf-8') as csvFile:
                np.savetxt(csvFile, current_charge, delimiter=',', fmt='%.4f')


if __name__ == '__main__':
    file = file_name('./rawData')
    for fileName in file:
        filePath = './rawData/' + fileName
        name = fileName[-9:-4]
        if not os.path.exists('./DataSet/' + name):
            os.mkdir('./DataSet/' + name)
        get_data(filePath, name)
        print('成功获取%s的放电数据！成功提取%s的健康因子！' % (fileName, fileName))
