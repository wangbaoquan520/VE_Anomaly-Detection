import numpy as np
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist, pdist, squareform

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
        allTracks[stationID] = tracks
        allDateList[stationID] = datelist
    return allTracks, allDateList


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


if __name__ == '__main__':
    fname = 'kmeans'
    allfilePath = "./data/allStationsFromOracl.csv"
    testfilePath = "./data/allStations-all.csv"
    clusterOutFilePath = './cluster/%s_out.csv' % fname
    DBIfilePath = './cluster/%s_DBI.csv' % fname
    allTracks, allDateList = loadTrackData(allfilePath)  # 返回的为字典
    testTracks, testDateList = loadTrackData(testfilePath)
    clusters_n = 3
    iteration_n = 100
    positiveLabels = [0, 1, 2, 3, 4, 5, 6]
    threshold = 0.3
    stations = ['13303']
    # 训练模型
    for stationID in stations:
        print("stationID:", stationID)
        tracks = allTracks[stationID]
        dateList = allDateList[stationID]
        # codebook, distortion = vq.kmeans(obs=wightened, k_or_guess=k, iter=20, thresh=1e-05, check_finite=True)
        centroid, labels, inertia, best_n = k_means(tracks, n_clusters=clusters_n, n_init=clusters_n, max_iter=100,
                                                    return_n_iter=True)
        f = open('./kmeansOut/' + stationID + ".csv", "a+")  # 打开文件以便写入
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
        f = open('./kmeansOut/' + "allStationIDs.csv", "a+")  # 打开文件以便写入
        print(stationID, file=f)
        f.close()
        # with open('./kmeansOut/' % fname, "a") as f:
        #     # f.write(str(stationID) + "," + str(metrics.davies_bouldin_score(wightened, labels)))
        #     f.write(str(labels) + '\n')



        # 参数 labels,tracks,trainLabels
        # trainTracks, trainLabels = gettrainLabelsData(labels, tracks, positiveLabels)
        # if 21 > len(trainTracks):
        #     print("batch_size is bigger than N")
        #     continue
        # if not os.path.exists(ckptPath + stationID):
        #     os.makedirs(ckptPath + stationID)
        # if not os.path.exists(summaries_dir + stationID):
        #     os.makedirs(summaries_dir + stationID)
        # 参数 trainOneModel(trainTracks,trainLabels,summaries_dir,stationID,ckptPath)
        # trainOneModel(trainTracks,trainLabels,summaries_dir+stationID,stationID,ckptPath + stationID +"//modelsave.ckpt",testTracks, testDateList)
    # 测试模型
    # for stationID in testTracks:
    #     print("stationID:", stationID)
    #     if not os.path.exists(ckptPath + stationID):
    #         print("no model")
    #         continue
    #     if not os.path.exists(summaries_dir + stationID):
    #         print("no model")
    #         continue
    #     tracks = testTracks[stationID]
    #     dateList = testDateList[stationID]
    #     # 参数 dectOutlier(stationID, dataInstance, dateList, trainLabels, ckptPath)
    #     dectOutlier(stationID,tracks,dateList,positiveLabels,ckptPath)








