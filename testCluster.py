# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import canopy
from sklearn.cluster import DBSCAN,k_means
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# coding=utf-8

import numpy as np


class KMeans(object):
    """
    - 参数
        n_clusters:
            聚类个数，即k
        initCent:
            质心初始化方式，可选"random"或指定一个具体的array,默认random，即随机初始化
        max_iter:
            最大迭代次数
    """

    def __init__(self, n_clusters=5, initCent='random', max_iter=300):
        if hasattr(initCent, '__array__'):
            n_clusters = initCent.shape[0]
            self.centroids = np.asarray(initCent, dtype=np.float)
        else:
            self.centroids = None

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.initCent = initCent
        self.clusterAssment = None
        self.labels = None
        self.sse = None

        # 计算两点的欧式距离

    def _distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    # 随机选取k个质心,必须在数据集的边界内
    def _randCent(self, X, k):
        n = X.shape[1]  # 特征维数
        centroids = np.empty((k, n))  # k*n的矩阵，用于存储质心
        for j in range(n):  # 产生k个质心，一维一维地随机初始化
            minJ = min(X[:, j])
            rangeJ = float(max(X[:, j]) - minJ)
            centroids[:, j] = (minJ + rangeJ * np.random.rand(k, 1)).flatten()
        return centroids

    def fit(self, X):
        # 类型检查
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]  # m代表样本数量
        self.clusterAssment = np.empty((m, 2))  # m*2的矩阵，第一列存储样本点所属的族的索引值，
        # 第二列存储该点与所属族的质心的平方误差
        if self.initCent == 'random':
            self.centroids = self._randCent(X, self.n_clusters)

        clusterChanged = True
        for _ in range(self.max_iter):
            clusterChanged = False
            for i in range(m):  # 将每个样本点分配到离它最近的质心所属的族
                minDist = np.inf;
                minIndex = -1
                for j in range(self.n_clusters):
                    distJI = self._distEclud(self.centroids[j, :], X[i, :])
                    if distJI < minDist:
                        minDist = distJI;
                        minIndex = j
                if self.clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                    self.clusterAssment[i, :] = minIndex, minDist ** 2

            if not clusterChanged:  # 若所有样本点所属的族都不改变,则已收敛，结束迭代
                break
            for i in range(self.n_clusters):  # 更新质心，即将每个族中的点的均值作为质心
                ptsInClust = X[np.nonzero(self.clusterAssment[:, 0] == i)[0]]  # 取出属于第i个族的所有点
                self.centroids[i, :] = np.mean(ptsInClust, axis=0)

        self.labels = self.clusterAssment[:, 0]
        self.sse = sum(self.clusterAssment[:, 1])

    def predict(self, X):  # 根据聚类结果，预测新输入数据所属的族
        # 类型检查
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]  # m代表样本数量
        preds = np.empty((m,))
        for i in range(m):  # 将每个样本点分配到离它最近的质心所属的族
            minDist = np.inf
            for j in range(self.n_clusters):
                distJI = self._distEclud(self.centroids[j, :], X[i, :])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds


class biKMeans(object):
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.centroids = None
        self.clusterAssment = None
        self.labels = None
        self.sse = None

    # 计算两点的欧式距离
    def _distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    def fit(self, X):
        m = X.shape[0]
        self.clusterAssment = np.zeros((m, 2))
        centroid0 = np.mean(X, axis=0).tolist()
        centList = [centroid0]
        for j in range(m):  # 计算每个样本点与质心之间初始的平方误差
            self.clusterAssment[j, 1] = self._distEclud(np.asarray(centroid0), X[j, :]) ** 2

        while (len(centList) < self.n_clusters):
            lowestSSE = np.inf
            for i in range(len(centList)):  # 尝试划分每一族,选取使得误差最小的那个族进行划分
                ptsInCurrCluster = X[np.nonzero(self.clusterAssment[:, 0] == i)[0], :]
                clf = KMeans(n_clusters=2)
                clf.fit(ptsInCurrCluster)
                centroidMat, splitClustAss = clf.centroids, clf.clusterAssment  # 划分该族后，所得到的质心、分配结果及误差矩阵
                sseSplit = sum(splitClustAss[:, 1])
                sseNotSplit = sum(self.clusterAssment[np.nonzero(self.clusterAssment[:, 0] != i)[0], 1])
                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            # 该族被划分成两个子族后,其中一个子族的索引变为原族的索引，另一个子族的索引变为len(centList),然后存入centList
            bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centList)
            bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit
            centList[bestCentToSplit] = bestNewCents[0, :].tolist()
            centList.append(bestNewCents[1, :].tolist())
            self.clusterAssment[np.nonzero(self.clusterAssment[:, 0] == bestCentToSplit)[0], :] = bestClustAss

        self.labels = self.clusterAssment[:, 0]
        self.sse = sum(self.clusterAssment[:, 1])
        self.centroids = np.asarray(centList)

    def predict(self, X):  # 根据聚类结果，预测新输入数据所属的族
        # 类型检查
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]  # m代表样本数量
        preds = np.empty((m,))
        for i in range(m):  # 将每个样本点分配到离它最近的质心所属的族
            minDist = np.inf
            for j in range(self.n_clusters):
                distJI = self._distEclud(self.centroids[j, :], X[i, :])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds


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
        # tracks = (tracks - fillNumsmin) / (fillNumsmax - fillNumsmin)  # 归一化
        # tracks = tracks + 1
        allTracks[stationID] = tracks
        allDateList[stationID] = datelist
    return allTracks, allDateList

def compute_distances_no_loops(X):
    dists = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        dists[i] = np.sqrt(np.sum(np.square(X - X[i]), axis=1))
    return dists

def db_index(X, y):
    """
    Davies-Bouldin index is an internal evaluation method for
    clustering algorithms. Lower values indicate tighter clusters that
    are better separated.
    """
    # get unique labels
    if y.ndim == 2:
        y = np.argmax(axis=1)
    uniqlbls = np.unique(y)
    n = len(uniqlbls)
    # pre-calculate centroid and sigma
    centroid_arr = np.empty((n, X.shape[1]))
    sigma_arr = np.empty((n,1))
    dbi_arr = np.empty((n,n))
    mask_arr = np.invert(np.eye(n, dtype='bool'))
    for i,k in enumerate(uniqlbls):
        Xk = X[np.where(y==k)[0],...]
        Ak = np.mean(Xk, axis=0)
        centroid_arr[i,...] = Ak
        sigma_arr[i,...] = np.mean(cdist(Xk, Ak.reshape(1,-1)))
    # compute pairwise centroid distances, make diagonal elements non-zero
    centroid_pdist_arr = squareform(pdist(centroid_arr)) + np.eye(n)
    # compute pairwise sigma sums
    sigma_psum_arr = squareform(pdist(sigma_arr, lambda u,v: u+v))
    # divide
    dbi_arr = np.divide(sigma_psum_arr, centroid_pdist_arr)
    # get mean of max of off-diagonal elements
    dbi_arr = np.where(mask_arr, dbi_arr, 0)
    dbi = np.mean(np.max(dbi_arr, axis=1))
    return dbi

direc = './UCR_TS_Archive_2015'

def main():
    allfilePath = "./data/allStationsFromOracl.csv"
    testfilePath = "./data/allStations-all.csv"
    allTracks, allDateList = loadTrackData(allfilePath)  # 返回的为字典
    del allTracks["13312"]
    testTracks, testDateList = loadTrackData(testfilePath)
    # 设置画布的大小
    pig = plt.figure(figsize=(12, 6))
    # markers = ['4',".","^","2","s","*","d","+","P"]
    # colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
    colors = ['brown', 'green', 'blue', 'y', 'r', 'tan', 'dodgerblue', 'deeppink', 'orangered', 'olive', 'lime',
              'slategrey',
              'gold', 'dimgray', 'darkorange', 'peru', 'cyan', 'orchid', 'sienna', 'slateblue', 'thistle', 'violet',
              'lightgreen'
        , 'deepskyblue', 'cadetblue', 'lightblue', 'peachpuff', 'lightsalmon', 'skyblue', 'teal']
    markers = ['*', 'h', 'H', '+', 'o', '1', '2', '3', ',', 'v', 'H', '+', '1', '2', '^',
               '<', '>', '.', '4', 'H', '+', '1', '2', 's', 'p', 'x', 'D', 'd', '|', '_']
    # -----------------------------------------------  k-means  -------------------------------------------------
    # print('-----------------------------------------------  k-means  -------------------------------------------------')
    # for stationID in allTracks:
    #     fname = 'k-means'
    #     print("stationID:", stationID)
    #     tracks = allTracks[stationID]
    #     dateList = allDateList[stationID]
    #     centroid,labels,inertia,best_n= k_means(tracks,n_clusters = 8, n_init=10, max_iter=100,return_n_iter=True)
    #
    #     # codebook,distortion = vq.kmeans(obs = wightened, k_or_guess = 9, iter = 20, thresh = 1e-05, check_finite = True)
    #     # labels = vq.vq(wightened, codebook)[0]
    #     with open('./clusterDBI/%s_DBI.csv' % fname, "a") as f:
    #         # f.write(str(stationID) + "," + str(metrics.davies_bouldin_score(wightened, labels)) + '\n')
    #         f.write(str(stationID) + "," + str(db_index(tracks, labels)) + '\n')
    #     # plt.figure(figsize=(10, 10))
    #     # lw = 2
    #     # plt.scatter(wightened[:, 0], wightened[:, 1])
    #     # plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    #     print(fname + ':' + str(len(np.unique(labels))))
    #     # plt.scatter(tracks[:, 0], tracks[:, 1], c=labels)
    #     # plt.tight_layout()
    #     # 使用TSNE进行降维处理
    #     tsne = TSNE(n_components=2, learning_rate=100).fit_transform(tracks)
    #     # 使用PCA 进行降维处理
    #     pca = PCA().fit_transform(tracks)
    #     plt.clf()
    #     plt.subplot(121)
    #     plt.scatter(tsne[:, 0], tsne[:, 1], c=labels)
    #     plt.subplot(122)
    #     plt.scatter(pca[:, 0], pca[:, 1], c=labels)
    #     plt.colorbar()
    #     plt.savefig("./cluster/"+str(stationID)+"_%s.png" % fname, dpi=100)
    #     with open("./cluster/"+str(stationID)+"_%s_labels.csv" % fname, "a") as f:
    #         for label in labels:
    #             f.write(str(label) + '\n')

    # -----------------------------------------------  canopy+k-means  -------------------------------------------------
    print('-----------------------------------------------  canopy+k-means  -------------------------------------------------')
    for stationID in allTracks:
        f = open('./kmeansOut/' + "allStationIDs.csv", "a+")  # 打开文件以便写入
        print(stationID, file=f)
        f.close()
        fname = 'canopy+k-means'
        print("stationID:", stationID)
        tracks = allTracks[stationID]
        dateList = allDateList[stationID]
        # wightened = vq.whiten(tracks)
        dm = compute_distances_no_loops(tracks)
        t1 = dm.max() / np.sqrt(2)
        t2 = dm.max() / 2
        gc = canopy.Canopy(tracks)
        gc.setThreshold(t1, t2)
        canopies = gc.clustering()
        if len(canopies) > 30:
            continue
        canopy.showCanopy(canopies,tracks,t1,t2,stationID,pig)
        k = len(canopies)
        print('canopy K:', str(k))
        with open('./clusterDBI/testK_%s_DBI.csv' % fname, "a") as f:
            # f.write(str(stationID) + "," + str(metrics.davies_bouldin_score(wightened, labels)))
            f.write('canopy K'+ ','+ str(k) + '\n')
        for testK in range(2,16):

            # codebook, distortion = vq.kmeans(obs=wightened, k_or_guess=k, iter=20, thresh=1e-05, check_finite=True)
            centroid, labels, inertia, best_n = k_means(tracks, n_clusters=testK, n_init=k, max_iter=100,
                                                        return_n_iter=True)
            with open('./clusterDBI/testK_%s_DBI.csv' % fname, "a") as f:
                # f.write(str(stationID) + "," + str(metrics.davies_bouldin_score(wightened, labels)))
                f.write(str(stationID) + "," + str(testK) + ',' + str(db_index(tracks, labels)) + '\n')
            # 质心与原数据合并 之后降维
            originLength = tracks.shape[0]
            allData = np.concatenate((tracks,centroid),axis=0)
            # 使用TSNE进行降维处理
            tsne = TSNE(n_components=2, learning_rate=100).fit_transform(allData)
            # 使用PCA 进行降维处理
            pca = PCA().fit_transform(allData)
            plt.clf()
            for i in range(testK):
                plt.title("SSE={:.2f}".format(inertia))
                index = np.nonzero(labels == i)[0]
                plt.subplot(121)
                x0 = tsne[index, 0]
                x1 = tsne[index, 1]
                for j in range(len(x0)):
                    plt.scatter(x0[j], x1[j], c=colors[i],marker = markers[i])
                plt.scatter(tsne[originLength+i, 0], tsne[originLength+i, 1], marker='x', color=colors[i],
                                                          linewidths=8)
                plt.subplot(122)
                x0 = pca[index, 0]
                x1 = pca[index, 1]
                for j in range(len(x0)):
                    plt.scatter(x0[j], x1[j], c=colors[i], marker=markers[i])
                plt.scatter(pca[originLength+i, 0], pca[originLength+i, 1], marker='x', color=colors[i],
                            linewidths=8)
            plt.savefig("./cluster/" + str(stationID) + "_%s_%s.png" % (fname,str(testK)), dpi=100)
            f = open('./kmeansOut/' + stationID+"_"+str(testK) + ".csv", "a+")  # 打开文件以便写入
            # print('$' + stationID, file=f)
            for index in range(len(dateList)):
                oneDayList = tracks[index]
                oneDayList = oneDayList.astype('str')
                # print("#" + dateList[index], file=f)
                indexFroData = 0
                for indexFroOneDay in range(24):
                    for indexForOneHour in range(2):
                        key = dateList[index] + str(indexFroOneDay).zfill(2) + str(indexForOneHour * 30).zfill(2)
                        print(key + ',' + oneDayList[indexFroData] + "," + str(labels[index]), file=f)
                        indexFroData = indexFroData + 1

                # for one in oneDayList:
                #     print(one + "," + str(labels[index]), file=f)
            f.close()



    # # -----------------------------------------------  dbscan  -------------------------------------------------
    # print('-----------------------------------------------  dbscan  -------------------------------------------------')
    # for stationID in allTracks:
    #     fname = 'dbscan'
    #     print("stationID:", stationID)
    #     tracks = allTracks[stationID]
    #     dateList = allDateList[stationID]
    #     # wightened = vq.whiten(tracks)
    #     dm = compute_distances_no_loops(tracks)
    #     t1 = dm.max() / np.sqrt(2)
    #     t2 = dm.max() / 20
    #     print("dm.max()" + str(dm.max()))
    #     print("dm.min()" + str(dm.min()))
    #     labels = DBSCAN(eps=0.001).fit_predict(tracks)
    #     print(fname + ':' +str(len(np.unique(labels))))
    #     # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    #     # plt.show()
    #     # codebook, distortion = vq.kmeans(obs=wightened, k_or_guess=9, iter=20, thresh=1e-05, check_finite=True)
    #     # labels = vq(wightened, codebook)[0]
    #     with open('./clusterDBI/%s_DBI.csv' % fname, "a") as f:
    #         # f.write(str(stationID) + "," + str(metrics.davies_bouldin_score(wightened, labels)))
    #         f.write(str(stationID) + "," + str(db_index(tracks, labels)) + '\n')
    #     # plt.figure(figsize=(10, 10))
    #     # lw = 2
    #     # plt.scatter(wightened[:, 0], wightened[:, 1])
    #     # plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    #     # plt.scatter(tracks[:, 0], tracks[:, 1], c=labels)
    #     # plt.tight_layout()
    #     # 使用TSNE进行降维处理
    #     tsne = TSNE(n_components=2, learning_rate=100).fit_transform(tracks)
    #     # 使用PCA 进行降维处理
    #     pca = PCA().fit_transform(tracks)
    #     plt.clf()
    #     plt.subplot(121)
    #     plt.scatter(tsne[:, 0], tsne[:, 1], c=labels)
    #     plt.subplot(122)
    #     plt.scatter(pca[:, 0], pca[:, 1], c=labels)
    #     plt.colorbar()
    #     plt.savefig("./cluster/" + str(stationID) + "_%s.png" % fname, dpi=100)
    #     with open("./cluster/"+str(stationID)+"_%s_labels.csv" % fname, "a") as f:
    #         for label in labels:
    #             f.write(str(label) + '\n')
    # # t1 = 0.6
    # # t2 = 0.4
    # # # 随机生成 500 个二维[0,1)平面点
    # # dataset = np.random.rand(500, 2)
    # # gc = Canopy(dataset)
    # # gc.setThreshold(t1, t2)
    # # canopies = gc.clustering()
    # # print('Get %s initial centers.' % len(canopies))
    # # showCanopy(canopies, dataset, t1, t2)


def plotKDistanceCurve():
    allfilePath = "./data/allStationsFromOracl.csv"
    testfilePath = "./data/allStations-all.csv"
    allTracks, allDateList = loadTrackData(allfilePath)  # 返回的为字典
    del allTracks["13312"]
    testTracks, testDateList = loadTrackData(testfilePath)
    for stationID in allTracks:
        print("stationID:", stationID)
        tracks = allTracks[stationID]
        dists = compute_distances_no_loops(tracks)
        for row in dists:
            temp = np.copy(row)
            temp = sorted(temp)


def quicksort(num, low, high):  # 快速排序
    if low < high:
        location = partition(num, low, high)
        quicksort(num, low, location - 1)
        quicksort(num, location + 1, high)


def partition(num, low, high):
    pivot = num[low]
    while (low < high):
        while (low < high and num[high] > pivot):
            high -= 1
        while (low < high and num[low] < pivot):
            low += 1
        temp = num[low]
        num[low] = num[high]
        num[high] = temp
    num[low] = pivot
    return low


def findkth(num, low, high, k):  # 找到数组里第k个数
    index = partition(num, low, high)
    if index == k: return num[index]
    if index < k:
        return findkth(num, index + 1, high, k)
    else:
        return findkth(num, low, index - 1, k)


pai = [2, 3, 1, 5, 4, 6]
# quicksort(pai, 0, len(pai) - 1)

print(findkth(pai, 0, len(pai) - 1, 0))

if __name__ == '__main__':
    main()