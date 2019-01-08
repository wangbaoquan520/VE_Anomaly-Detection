# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import tensorflow as tf #1.7

def kmeansForStation(points, dateList,clusters_n,iteration_n,outfilePath,stationID):
    plt.clf() # Clear the current figure.
    pointsInTensor = tf.convert_to_tensor(points)
    # tf.random_shuffle
    # 沿着要被洗牌的张量的第一个维度，随机打乱。
    # tf.slice(inputs,begin,size,name='')
    # 用途：从inputs中抽取部分内容
    #      inputs：可以是list,array,tensor
    #      begin：n维列表，begin[i] 表示从inputs中第i维抽取数据时，相对0的起始偏移量，也就是从第i维的begin[i]开始抽取数据
    #      size：n维列表，size[i]表示要抽取的第i维元素的数目 size[i]=-1 表示第i维从begin[i]剩余的元素都要被抽取
    # centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))
    centroids = tf.Variable(tf.slice(pointsInTensor, [0, 0], [clusters_n, -1]))  # 取前clusters_n 作为初始的中心

    points_expanded = tf.expand_dims(pointsInTensor, 0)
    centroids_expanded = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)  # 欧式距离的平方

    # 余弦距离
    # 求模
    # points_expanded_norm = tf.sqrt(tf.reduce_sum(tf.square(points_expanded), axis=2))
    # centroids_expanded_norm = tf.sqrt(tf.reduce_sum(tf.square(centroids_expanded), axis=2))
    # # 内积
    # x3_x4 = tf.reduce_sum(tf.multiply(points_expanded, centroids_expanded), axis=2)
    # distances = tf.divide(x3_x4, tf.multiply(points_expanded_norm, centroids_expanded_norm))

    assignments = tf.argmin(distances, 0)  # tf.argmin(input, dimension, name=None) 函數解說:沿著需要的維度找尋最小值的索引值,最小由0開始

    means = []
    for c in range(clusters_n):
        means.append(tf.reduce_mean(
            tf.gather(pointsInTensor,
                      tf.reshape(
                          tf.where(
                              tf.equal(assignments, c)
                          ), [1, -1])
                      ), reduction_indices=[1]))
    # tf.concat是连接两个矩阵的操作 tf.concat(concat_dim, values, name='concat')
    new_centroids = tf.concat(means, 0)
    # tf.assign(ref, value, validate_shape=None, use_locking=None, name=None)
    # 函数完成了将value赋值给ref的作用。其中：ref 必须是tf.Variable创建的tensor，如果ref=tf.constant()会报错！
    # 同时，shape（value）==shape（ref）
    update_centroids = tf.assign(centroids, new_centroids)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(iteration_n):
            [_, centroid_values, points_values, assignment_values] = sess.run(
                [update_centroids, centroids, pointsInTensor, assignments])

        # print("centroids" + "\n", centroid_values)

    plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
    # plt.scatter(points_values[:, 0], points_values[:, 1], c=np.squeeze(assignment_values), s=50, alpha=0.5)
    plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
    fig = plt.gcf()
    # plt.show()
    fig.savefig(outfilePath + stationID +".png", dpi=100)
    # plt.savefig(outfilePath+".jpg")
    labels = assignment_values.tolist()
    f = open(outfilePath+stationID+".csv", "a+")  # 打开文件以便写入
    # print('$' + stationID, file=f)
    for index in range(len(dateList)):
        oneDayList = points[index]
        oneDayList = oneDayList.astype('str')
        # print("#" + dateList[index], file=f)
        indexFroData = 0
        for indexFroOneDay in range(24):
            for indexForOneHour in range(2):
                key = dateList[index] + str(indexFroOneDay).zfill(2) + str(indexForOneHour * 30).zfill(2)
                print(key+','+ oneDayList[indexFroData] + "," + str(labels[index]), file=f)
                indexFroData = indexFroData +1

        # for one in oneDayList:
        #     print(one + "," + str(labels[index]), file=f)
    f.close()
    f = open(outfilePath  + "allStationIDs.csv", "a+")  # 打开文件以便写入
    print(stationID, file=f)
    f.close()
    print("kmeans is done！")
    return  labels

def kmeans(points,clusters_n,iteration_n,clusterOutFilePath,clusterJPGFilePath):
    plt.clf() # Clear the current figure.
    pointsInTensor = tf.convert_to_tensor(points)
    # tf.random_shuffle
    # 沿着要被洗牌的张量的第一个维度，随机打乱。
    # tf.slice(inputs,begin,size,name='')
    # 用途：从inputs中抽取部分内容
    #      inputs：可以是list,array,tensor
    #      begin：n维列表，begin[i] 表示从inputs中第i维抽取数据时，相对0的起始偏移量，也就是从第i维的begin[i]开始抽取数据
    #      size：n维列表，size[i]表示要抽取的第i维元素的数目 size[i]=-1 表示第i维从begin[i]剩余的元素都要被抽取
    # centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))
    centroids = tf.Variable(tf.slice(pointsInTensor, [0, 0], [clusters_n, -1]))  # 取前clusters_n 作为初始的中心

    points_expanded = tf.expand_dims(pointsInTensor, 0)
    centroids_expanded = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)  # 欧式距离的平方

    # 余弦距离
    # 求模
    # points_expanded_norm = tf.sqrt(tf.reduce_sum(tf.square(points_expanded), axis=2))
    # centroids_expanded_norm = tf.sqrt(tf.reduce_sum(tf.square(centroids_expanded), axis=2))
    # # 内积
    # x3_x4 = tf.reduce_sum(tf.multiply(points_expanded, centroids_expanded), axis=2)
    # distances = tf.divide(x3_x4, tf.multiply(points_expanded_norm, centroids_expanded_norm))

    assignments = tf.argmin(distances, 0)  # tf.argmin(input, dimension, name=None) 函數解說:沿著需要的維度找尋最小值的索引值,最小由0開始

    means = []
    for c in range(clusters_n):
        means.append(tf.reduce_mean(
            tf.gather(pointsInTensor,
                      tf.reshape(
                          tf.where(
                              tf.equal(assignments, c)
                          ), [1, -1])
                      ), reduction_indices=[1]))
    # tf.concat是连接两个矩阵的操作 tf.concat(concat_dim, values, name='concat')
    new_centroids = tf.concat(means, 0)
    # tf.assign(ref, value, validate_shape=None, use_locking=None, name=None)
    # 函数完成了将value赋值给ref的作用。其中：ref 必须是tf.Variable创建的tensor，如果ref=tf.constant()会报错！
    # 同时，shape（value）==shape（ref）
    update_centroids = tf.assign(centroids, new_centroids)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(iteration_n):
            [_, centroid_values, points_values, assignment_values] = sess.run(
                [update_centroids, centroids, pointsInTensor, assignments])
        # print("centroids" + "\n", centroid_values)
    plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
    # plt.scatter(points_values[:, 0], points_values[:, 1], c=np.squeeze(assignment_values), s=50, alpha=0.5)
    plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
    fig = plt.gcf()
    # plt.show()
    fig.savefig(clusterJPGFilePath, dpi=100)
    # plt.savefig(outfilePath+".jpg")
    labels = assignment_values.tolist()
    f = open(clusterOutFilePath, "a+")  # 打开文件以便写入
    print(labels, file=f)
    f.close()
    print("kmeans is done！")
    return  labels