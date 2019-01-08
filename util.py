# -*-coding:utf-8-*-
import numpy as np
import math
import pandas as pd
from csv import reader
import pickle
from sklearn.metrics import roc_auc_score

# 加载加油数据
# 返回结构：tracks[date]->list<fillNum>
def loadTrackData(filePath):
    stationID = ''
    allTracks = {}
    allDateList = {}
    tracks = []
    fillNums = set()
    dateTime = ''
    oneDayList = []
    datelist = []
    # max,min = 0.0
    with open(filePath) as f:
        for oneLine in f:
            oneLine = oneLine.strip('\n')
            if (oneLine.startswith("$")):
                if (len(tracks) > 0):
                    tracks = np.array(tracks)
                    fillNumsmax = max(fillNums)
                    fillNumsmin = min(fillNums)
                    if (fillNumsmax != 0):
                        # print("stationID" + stationID)
                        # print("fillNumsmax:" + str(fillNumsmax) + "   fillNumsmin:" + str(fillNumsmin))
                        tracks = (tracks - fillNumsmin) / (fillNumsmax - fillNumsmin)  # 归一化
                        allTracks[stationID] = tracks
                        allDateList[stationID] = datelist
                tracks = []
                fillNums = set()
                dateTime = ''
                oneDayList = []
                datelist = []
                stationID = oneLine[1:]
            elif oneLine.startswith("#"):
                # 存储前一天的数据
                if dateTime != '':
                    datelist.append(dateTime)
                    tracks.append(np.array(oneDayList).astype('float32'))
                dateTime = oneLine[1:]
                oneDayList = []  # 清空前一天的数据
            else:
                oneDayList.append((float)(oneLine.split(",")[1]))
                fillNums.add((float)(oneLine.split(",")[1]))  # 存储为了归一化
        # 存储最后一个加油站数据
        # fillNums = np.array(fillNums).astype('float32')
        tracks = np.array(tracks)
        fillNumsmax = max(fillNums)
        fillNumsmin = min(fillNums)
        tracks = (tracks - fillNumsmin) / (fillNumsmax - fillNumsmin)  # 归一化
        allTracks[stationID] = tracks
        allDateList[stationID] = datelist

    # tracks = []
    # fillNums = []
    # with open(filePath) as f:
    #     dateTime = ''
    #     oneDayList = []
    #     datelist = []
    #     for oneLine in f:
    #         oneLine = oneLine.strip('\n')
    #         # print(oneLine)
    #         if oneLine.startswith("#"):
    #             # 存储前一天的数据
    #             if dateTime != '':
    #                 datelist.append(dateTime)
    #                 tracks.append(np.array(oneDayList).astype('float32'))
    #             dateTime = oneLine
    #             oneDayList = [] #清空前一天的数据
    #         else:
    #             oneDayList.append(oneLine.split(",")[1])
    #             fillNums.append(oneLine.split(",")[1]) # 存储为了归一化
    #     # 存储最后一天的数据
    #     datelist.append(dateTime)
    #     tracks.append(np.array(oneDayList).astype('float32'))
    # # 归一化
    # fillNums = np.array(fillNums).astype('float32')
    # max = fillNums.max()
    # min = fillNums.min()
    # tracks = (tracks - min) / (max - min)
    # # for row in tracks:
    # #     row = (row - min) / (max - min)
    #     # tracks[date] = (tracks[date] - min) / (max - min)
    return allTracks, allDateList

# 加载加油数据
def loadTestTrackData(filePath):
    tracks = []
    fillNums = []
    with open(filePath) as f:
        dateTime = ''
        oneDayList = []
        datelist = []
        for oneLine in f:
            oneLine = oneLine.strip('\n')
            # print(oneLine)
            if oneLine.startswith("#"):
                # 存储前一天的数据
                if dateTime != '':
                    datelist.append(dateTime)
                    tracks.append(np.array(oneDayList).astype('float32'))
                dateTime = oneLine
                oneDayList = []  # 清空前一天的数据
            else:
                oneDayList.append(oneLine.split(",")[1])
                fillNums.append(oneLine.split(",")[1])  # 存储为了归一化
        # 存储最后一天的数据
        datelist.append(dateTime)
        tracks.append(np.array(oneDayList).astype('float32'))
    # 归一化
    fillNums = np.array(fillNums).astype('float32')
    max = fillNums.max()
    min = fillNums.min()
    tracks = (tracks - min) / (max - min)
    # for row in tracks:
    #     row = (row - min) / (max - min)
    # tracks[date] = (tracks[date] - min) / (max - min)
    return tracks, datelist

def load(filePath):
    # w = pd.read.csv(filePath)
    # w.describe()
    #
    # with open(filePath, 'rt', encoding='UTF-8')as raw_data:
    #     readers = reader(raw_data, delimiter=',')
    #     x = list(readers)
    #     data = np.array(x)
    #     print(data)
    #     print(data.shape)
    with open(filePath, 'rt', encoding='UTF-8') as raw_data:
        data = np.loadtxt(raw_data, delimiter=',')
    # print(data.shape)
    return data


# 使用欧式距离进行距离的计算
def euclideanDistance(self, vec1, vec2):
    return math.sqrt(((vec1 - vec2) ** 2).sum())


# X = pickle.load(open("input/breast_cancerX", "rb"))
# Y = pickle.load(open("input/breast_cancerY", "rb"))

def analyze_outlier(df):
    sorted_df = df.sort_values(by=['OF'], ascending=False)
    first30 = sorted_df.head(30)
    first30_Y = first30['Y'].tolist()
    first30_0count = first30_Y.count(0)
    first30_accuracy = (first30_0count / 30) * 100

    print(
        "Within the top 30 ranked cases (ranked according to the Outlier Factor), {} of the malignant cases (the outliers), comprising {}% of all malignant cases, were identified.".format(
            first30_0count, first30_accuracy))


def calculate_roc_auc(y_true, y_pred):
    print("ROC AUC score: {}".format(roc_auc_score(y_true, y_pred)))

if __name__ == '__main__':
    allfilePath = "D://GCN实验//data//allStations.csv"
    # a = set()
    # a.add(1)
    # a.add(2)
    # print(max(a))
    allTracks, allDateList = loadTrackData(allfilePath)
    print("ddddd")