# -*- coding: utf-8 -*-

import math
import random
import numpy as np
from datetime import datetime
from pprint import pprint as p
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Canopy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.t1 = 0
        self.t2 = 0

    # 设置初始阈值
    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print('t1 needs to be larger than t2!')

    # 使用欧式距离进行距离的计算
    def euclideanDistance(self, vec1, vec2):
        return math.sqrt(((vec1 - vec2) ** 2).sum())

    # 根据当前 dataset 的长度随机选择一个下标
    def getRandIndex(self,indexs):
        return random.randint(0, len(indexs) - 1)

    def clustering(self):
        if self.t1 == 0:
            print('Please set the threshold.')
        else:
            canopyIndexs = []
            canopiesInIndex = []
            canopies = []  # 用于存放最终归类结果
            indexs = range(len(self.dataset))
            while len(indexs) != 0:
                rand_index = self.getRandIndex(indexs)
                current_center_index = indexs[rand_index]
                current_center = self.dataset[indexs[rand_index]]  # 随机获取一个中心点，定为 P 点
                canopiesInIndex.append(indexs[rand_index])
                current_center_list = []  # 初始化 P 点的 canopy 类容器
                delete_list = []  # 初始化 P 点的删除容器
                indexs = np.delete(
                    indexs, rand_index, 0)  # 删除随机选择的中心点 P
                for datum_j in range(len(indexs)):
                    datum = self.dataset[indexs[datum_j]]
                    distance = self.euclideanDistance(
                        current_center, datum)  # 计算选取的中心点 P 到每个点之间的距离
                    if distance < self.t1:
                        # 若距离小于 t1，则将点归入 P 点的 canopy 类
                        current_center_list.append(indexs[datum_j])
                    if distance < self.t2:
                        delete_list.append(datum_j)  # 若小于 t2 则归入删除容器
                # 根据删除容器的下标，将元素从数据集中删除
                indexs = np.delete(indexs, delete_list, 0)
                canopies.append((current_center_index, current_center_list))
            # while len(self.dataset) != 0:
            #     rand_index = self.getRandIndex()
            #     current_center = self.dataset[rand_index]  # 随机获取一个中心点，定为 P 点
            #     current_center_list = []  # 初始化 P 点的 canopy 类容器
            #     delete_list = []  # 初始化 P 点的删除容器
            #     self.dataset = np.delete(
            #         self.dataset, rand_index, 0)  # 删除随机选择的中心点 P
            #     for datum_j in range(len(self.dataset)):
            #         datum = self.dataset[datum_j]
            #         distance = self.euclideanDistance(
            #             current_center, datum)  # 计算选取的中心点 P 到每个点之间的距离
            #         if distance < self.t1:
            #             # 若距离小于 t1，则将点归入 P 点的 canopy 类
            #             current_center_list.append(datum)
            #         if distance < self.t2:
            #             delete_list.append(datum_j)  # 若小于 t2 则归入删除容器
            #     # 根据删除容器的下标，将元素从数据集中删除
            #     self.dataset = np.delete(self.dataset, delete_list, 0)
            #     canopies.append((current_center, current_center_list))
        return canopies


def showCanopy(canopies, dataset, t1, t2,stationID,fig):
    fname = 'canopy'
    # fig = plt.figure()
    colors = ['brown', 'green', 'blue', 'y', 'r', 'tan', 'dodgerblue', 'deeppink', 'orangered','olive','lime','slategrey',
              'gold', 'dimgray', 'darkorange', 'peru', 'cyan',  'orchid', 'sienna','slateblue','thistle','violet','lightgreen'
              ,'deepskyblue','cadetblue','lightblue','peachpuff','lightsalmon','skyblue','teal']
    markers = ['*', 'h', 'H', '+', 'o', '1', '2', '3', ',', 'v', 'H', '+', '1', '2', '^',
               '<', '>', '.', '4', 'H', '+', '1', '2', 's', 'p', 'x', 'D', 'd', '|', '_']
    # 使用TSNE进行降维处理
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(dataset)
    # 使用PCA 进行降维处理
    pca = PCA().fit_transform(dataset)
    plt.clf()
    # plt.subplot(121)
    sc = fig.add_subplot(121)
    for i in range(len(canopies)):
        canopy = canopies[i]
        center = tsne[canopy[0]]
        components = canopy[1]
        sc.plot(center[0], center[1], marker=markers[i],
                color=colors[i], markersize=14)
        # t1_circle = plt.Circle(
        #     xy=(center[0], center[1]), radius=t1, color='dodgerblue', fill=False)
        # t2_circle = plt.Circle(
        #     xy=(center[0], center[1]), radius=t2, color='skyblue', alpha=0.2)
        # sc.add_artist(t1_circle)
        # sc.add_artist(t2_circle)
        for component in components:
            sc.plot((tsne[component])[0], (tsne[component])[1],
                    marker=markers[i], color=colors[i], markersize=1.5)
    # maxvalue = np.amax(tsne)
    # minvalue = np.amin(tsne)
    # plt.xlim(minvalue - t1, maxvalue + t1)
    # plt.ylim(minvalue - t1, maxvalue + t1)

    sc = fig.add_subplot(122)
    for i in range(len(canopies)):
        canopy = canopies[i]
        center = pca[canopy[0]]
        components = canopy[1]
        sc.plot(center[0], center[1], marker=markers[i],
                color=colors[i], markersize=30)
        # t1_circle = plt.Circle(
        #     xy=(center[0], center[1]), radius=t1, color='dodgerblue', fill=False)
        # t2_circle = plt.Circle(
        #     xy=(center[0], center[1]), radius=t2, color='skyblue', alpha=0.2)
        # sc.add_artist(t1_circle)
        # sc.add_artist(t2_circle)
        for component in components:
            sc.plot((pca[component])[0], (pca[component])[1],
                    marker=markers[i], color=colors[i], markersize=1.5)
    # maxvalue = np.amax(pca)
    # minvalue = np.amin(pca)
    # plt.xlim(minvalue - t1, maxvalue + t1)
    # plt.ylim(minvalue - t1, maxvalue + t1)
    plt.savefig("./cluster/" + str(stationID) + "_%s_%s.png" % (fname, str(len(canopies))), dpi=100)


def main():
    t1 = 0.6
    t2 = 0.4
    # 随机生成 500 个二维[0,1)平面点
    dataset = np.random.rand(500, 2)
    gc = Canopy(dataset)
    gc.setThreshold(t1, t2)
    canopies = gc.clustering()
    print('Get %s initial centers.' % len(canopies))
    showCanopy(canopies, dataset, t1, t2)


if __name__ == '__main__':
    t_s = datetime.now()
    main()
    t_e = datetime.now()
    usedtime = t_e - t_s
    print('[%s]' % usedtime)