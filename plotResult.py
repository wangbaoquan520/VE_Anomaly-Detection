# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np

def plotTrainingTime():
    x = ['ElectricDevices', 'Two_Patterns', 'StarLightCurves', 'ECG5000']
    y_lstm = []
    y_rnn = []
    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(x, y_lstm, color='darkorange', lw=lw, linestyle='--')
    plt.plot(x, y_rnn, color='navy', lw=lw, linestyle=':')

    # plt.xlim([np.max(outLierF) * 0.40, maxoutlierThreshold * 1.1])
    # plt.ylim([0.2, 1.05])
    plt.xlabel('data set')
    plt.ylabel('Training Time')
    plt.title('Training time of data set')
    plt.tight_layout()
    plt.savefig("./plotResult/TrainingTime.png", dpi=100)


def plotDBI():
    plt.figure(figsize=(20, 12))
    lw = 4
    DBIA = [1.880151408,2.661326083,1.746270433,1.485746828,0.83339538,1.223879254,1.48518477,1.344918722,1.544961875,
            1.384450346,1.468037306,1.16979324,1.18785671,1.229221223]
    DBIB = [0.910065507,1.107748844,0.929302406,0.840649401,0.780203987,1.042230626,0.977750675,0.920921769,0.861471328,
            0.766874563,1.044177754,0.59698756,0.921368427,0.76119387]
    x = range(2,16)

    # 使用红色-星状标记需要绘制的点
    plt.plot(x, DBIA, 'r*', lw=lw, color='darkorange', linestyle='--',label='stationA')
    plt.plot(x, DBIB, 'r*', lw=lw, color='navy', linestyle=':',label='stationB')

    # # 将数组中的前两个点进行连线
    # plt.plot(x[:2], y[:2])

    # plt.xlim([np.max(outLierF) * 0.40, maxoutlierThreshold * 1.1])
    # plt.ylim([0.2, 1.05])
    plt.xlim(1, 16)
    plt.ylim(0, 5)
    # plt.xlabel('cluster num')
    # plt.ylabel('DBI')
    plt.xlabel('cluster num', fontsize=50)
    plt.ylabel('DBI', fontsize=50)
    plt.tick_params(labelsize=48)
    plt.legend(loc="upper right", fontsize=50)
    plt.grid(linestyle='-.',linewidth=2)
    plt.tight_layout()
    plt.savefig("./plotResult/DBI.png", dpi=100)


def plotROC1():
    flist = [ 'ElectricDevices','Two_Patterns','StarLightCurves','ECG5000']
    methods = ['LSTM-VE','RNN-RS','FCN','ResNet']
    plt.figure(figsize=(20, 20))
    lw = 2
    colors = ['darkorange','limegreen','royalblue','deeppink']
    markers = ['-','--','-.',':']
    for i in range(0,4):
        fname = flist[i]
        plt.subplot(221+i)
        for j in range(0,4):
            methodName = methods[j]
            color = colors[j]
            marker = markers[j]
            lable = []
            outlierF = []
            with open("./outlierF/%s_%s.csv" % (fname, methodName), "r") as file:
                for line in file.readlines():
                    line = line.strip('\n')
                    line =float(line)
                    outlierF.append(line)
            with open("./outlierF/%s_%s_labels.csv" % (fname, methodName), "r") as file:
                for line in file.readlines():
                    line = line.strip('\n')
                    line = int(line)
                    lable.append(line)
            fpr, tpr, threshold = roc_curve(lable, outlierF)
            roc_auc = auc(fpr, tpr)  ###计算auc的值
            print("%s   %s    AUC:"%(fname,methodName), roc_auc)
            plt.plot(fpr, tpr, color=color,
                     lw=lw,linestyle=marker, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate',fontsize=24)
            plt.ylabel('True Positive Rate',fontsize=24)
            plt.legend(loc="lower right",fontsize=24)
            plt.tick_params(labelsize=23)
    plt.tight_layout()
    plt.savefig("./plotResult/ROC.png", dpi=100)

def plotROC2():
    flist = [ 'ElectricDevices','Two_Patterns','StarLightCurves','ECG5000']
    flist = [ 'ECG5000']
    methods = ['LSTM-VE','RNN-RS','FCN','ResNet']
    lables = ['LSTM-VE','AE-RS','FCN-VE','ResNet-VE']
    plt.figure(figsize=(20, 20))
    lw = 4
    colors = ['darkorange','limegreen','royalblue','deeppink']
    markers = ['-', '--', '-.', ':']
    ns = [6,2,3,4]
    for i in range(len(flist)):
        plt.clf()
        fname = flist[i]
        n = ns[3]
        outlierFMax = (n-1) / (n * n)

        methodName = 'LSTM-VE'
        color = 'darkorange'
        marker = '-'
        legend = 'LSTM-VE'
        lable = []
        outlierF = []
        with open("./outlierF/%s_%s.csv" % (fname, methodName), "r") as file:
            for line in file.readlines():
                line = line.strip('\n')
                line = float(line)
                outlierF.append(line)
        with open("./outlierF/%s_%s_labels.csv" % (fname, methodName), "r") as file:
            for line in file.readlines():
                line = line.strip('\n')
                line = int(line)
                lable.append(line)
        for inde in range(len(lable)):
            if lable[inde] == 1:
                outlierF[inde] = outlierF[inde] / outlierFMax
            else:
                outlierF[inde] = 1 - outlierF[inde] / outlierFMax
        fpr, tpr, threshold = roc_curve(lable, outlierF)
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        print("%s   %s    AUC:" % (fname, methodName), roc_auc)
        plt.plot(fpr, tpr, color=color,
                 lw=lw, linestyle=marker, label='%s' %legend)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=50)
        plt.ylabel('True Positive Rate', fontsize=50)
        plt.legend(loc="lower right", fontsize=50)
        plt.tick_params(labelsize=48)

        methodName = 'RNN-RS'
        color = 'limegreen'
        marker = '-'
        legend = 'AE-RS'
        lable = []
        outlierF = []
        with open("./outlierF/%s_%s.csv" % (fname, methodName), "r") as file:
            for line in file.readlines():
                line = line.strip('\n')
                line = float(line)
                outlierF.append(line)
        with open("./outlierF/%s_%s_labels.csv" % (fname, methodName), "r") as file:
            for line in file.readlines():
                line = line.strip('\n')
                line = int(line)
                lable.append(line)
        # for inde in range(len(lable)):
        #     if lable[inde] == 1:
        #         outlierF[inde] = outlierF[inde] / max(outlierF)
        #     else:
        #         outlierF[inde] = 1 - outlierF[inde] / max(outlierF)
        for inde in range(len(lable)):
            if lable[inde] == 1:
                outlierF[inde] = (outlierF[inde] / max(outlierF))
            else:
                outlierF[inde] = (1 - outlierF[inde] / max(outlierF))
        fpr, tpr, threshold = roc_curve(lable, outlierF)
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        print("%s   %s    AUC:" % (fname, methodName), roc_auc)
        plt.plot(fpr, tpr, color=color,
                 lw=lw, linestyle=marker, label='%s' %legend)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=50)
        plt.ylabel('True Positive Rate', fontsize=50)
        plt.legend(loc="lower right", fontsize=50)
        plt.tick_params(labelsize=48)

        methodName = 'FCN'
        color = 'royalblue'
        marker = '-.'
        legend = 'FCN-VE'
        lable = []
        outlierF = []
        with open("./outlierF/%s_%s.csv" % (fname, methodName), "r") as file:
            for line in file.readlines():
                line = line.strip('\n')
                line = float(line)
                outlierF.append(line)
        with open("./outlierF/%s_%s_labels.csv" % (fname, methodName), "r") as file:
            for line in file.readlines():
                line = line.strip('\n')
                line = int(line)
                lable.append(line)
        for inde in range(len(lable)):
            if lable[inde] == 1:
                outlierF[inde] = outlierF[inde] / outlierFMax
            else:
                outlierF[inde] = 1 - outlierF[inde] / outlierFMax
        fpr, tpr, threshold = roc_curve(lable, outlierF)
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        print("%s   %s    AUC:" % (fname, methodName), roc_auc)
        plt.plot(fpr, tpr, color=color,
                 lw=lw, linestyle=marker, label='%s' %legend)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=50)
        plt.ylabel('True Positive Rate', fontsize=50)
        plt.legend(loc="lower right", fontsize=50)
        plt.tick_params(labelsize=48)

        methodName = 'ResNet'
        color = 'deeppink'
        marker = ':'
        legend = 'ResNet-VE'
        lable = []
        outlierF = []
        with open("./outlierF/%s_%s.csv" % (fname, methodName), "r") as file:
            for line in file.readlines():
                line = line.strip('\n')
                line = float(line)
                outlierF.append(line)
        with open("./outlierF/%s_%s_labels.csv" % (fname, methodName), "r") as file:
            for line in file.readlines():
                line = line.strip('\n')
                line = int(line)
                lable.append(line)
        for inde in range(len(lable)):
            if lable[inde] == 1:
                outlierF[inde] = outlierF[inde] / outlierFMax
            else:
                outlierF[inde] = 1 - outlierF[inde] / outlierFMax
        fpr, tpr, threshold = roc_curve(lable, outlierF)
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        print("%s   %s    AUC:" % (fname, methodName), roc_auc)
        plt.plot(fpr, tpr, color=color,
                 lw=lw, linestyle=marker, label='%s' %legend)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=50)
        plt.ylabel('True Positive Rate', fontsize=50)
        plt.legend(loc="lower right", fontsize=50)
        plt.tick_params(labelsize=48)

        plt.tight_layout()
        plt.savefig("./plotResult/%sROC.png"%fname, dpi=100)


def plotRS():
    flist = [ 'ElectricDevices','Two_Patterns','StarLightCurves','ECG5000']
    plt.figure(figsize=(20, 20))
    lw = 4
    for i in range(0, 4):
        plt.clf()
        fname = flist[i]
        methodName = 'RNN-RS'
        color = 'darkorange'
        outlierF = []
        with open("./outlierF/%s_%s.csv" % (fname, methodName), "r") as file:
            for line in file.readlines():
                line = line.strip('\n')
                line = float(line)
                outlierF.append(line)
        plt.plot(range(len(outlierF)), outlierF, color=color, lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('Index', fontsize=50)
        plt.ylabel('Reconstruction Error', fontsize=50)
        plt.tick_params(labelsize=48)
        plt.tight_layout()
        plt.savefig("./plotResult/%sRS.png"%fname, dpi=100)

def plotRSonFour():
    flist = ['ElectricDevices', 'Two_Patterns', 'StarLightCurves', 'ECG5000']
    colors = ['darkorange', 'limegreen', 'royalblue', 'deeppink']
    markers = ['-', '--', '-.', ':']
    plt.figure(figsize=(20, 10))
    lw = 4
    maxs = []
    mins = []
    for i in range(0, 4):
        plt.clf()
        fname = flist[i]
        methodName = 'RNN-RS'
        outlierF = []
        with open("./outlierF/%s_%s.csv" % (fname, methodName), "r") as file:
            for line in file.readlines():
                line = line.strip('\n')
                line = float(line)
                outlierF.append(line)
            maxs.append(max(outlierF))
            mins.append(min(outlierF))
    scale_ls = range(4)
    plt.plot(scale_ls, maxs, color=colors[0], lw=lw, linestyle=markers[0],label='max')
    plt.plot(scale_ls, mins, color=colors[1], lw=lw, linestyle=markers[1],label='min')
    plt.xticks(scale_ls, flist)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('dataset', fontsize=50)
    plt.ylabel('Reconstruction Error', fontsize=50)
    plt.legend(loc="upper right", fontsize=50)
    plt.grid(linestyle='-.',linewidth=2)
    plt.tick_params(labelsize=48)
    plt.tight_layout()
    plt.savefig("./plotResult/RSonFour.png", dpi=100)


def formData():
    labels1 = np.array(pd.read_csv('./12066/12066_6.csv',usecols=[2]))
    # print('labels')
    datas1 = np.array(pd.read_csv('./12066/12066_6.csv',usecols=[1]))
    # print('datas')

    labels = []
    for k in labels1:
        for j in k:
            labels.append(j)
    datas = []
    for k in datas1:
        for j in k:
            datas.append(j)

    data_0 = {}
    data_1 = {}
    data_2 = {}
    data_3 = {}
    data_4 = {}
    data_5 = {}
    for index in range(int(len(labels) / 48) + 1):
        label = labels[index*48]
        if label == 0:
            data_0[len(data_0)] = (datas[index:index+48])
        if label == 1:
            data_1[len(data_1)] = (datas[index:index+48])
        if label == 2:
            data_2[len(data_2)] = (datas[index:index+48])
        if label == 3:
            data_3[len(data_3)] = (datas[index:index+48])
        if label == 4:
            data_4[len(data_4)] = (datas[index:index+48])
        if label == 5:
            data_5[len(data_5)] = (datas[index:index+48])
        # index = index+48
    f0 = open('./12066/0.csv', 'a')
    f1 = open('./12066/1.csv', 'a')
    f2 = open('./12066/2.csv', 'a')
    f3 = open('./12066/3.csv', 'a')
    f4 = open('./12066/4.csv', 'a')
    f5 = open('./12066/5.csv', 'a')
    for i in range(48):
        for col in range(len(data_0)-1):
            f0.write(str(data_0[col][i]) + ',')
        f0.write(str(data_0[len(data_0)-1][i]) + '\n')

        for col in range(len(data_1)-1):
            f1.write(str(data_1[col][i]) + ',')
        f1.write(str(data_1[len(data_1)-1][i]) + '\n')

        for col in range(len(data_2)-1):
            f2.write(str(data_2[col][i]) + ',')
        f2.write(str(data_2[len(data_2)-1][i]) + '\n')

        for col in range(len(data_3)-1):
            f3.write(str(data_3[col][i]) + ',')
        f3.write(str(data_3[len(data_3)-1][i]) + '\n')

        for col in range(len(data_4)-1):
            f4.write(str(data_4[col][i]) + ',')
        f4.write(str(data_4[len(data_4)-1][i]) + '\n')

        for col in range(len(data_5)-1):
            f5.write(str(data_5[col][i]) + ',')
        f5.write(str(data_5[len(data_5)-1][i]) + '\n')
    f0.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()


if __name__ == '__main__':
    # plotDBI()
    # plotROC1()
    plotROC2()
    # plotRS()
    # plotRSonFour()
    # formData()