# -*-coding:utf-8-*-

#导包
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import mixture
from sklearn.svm.libsvm import predict

class DBSCAN:
    # 产生数据
    def create_data(centers, num=100, std=0.7):
        X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
        return X, labels_true

    """
        数据作图
    """

    def plot_data(*data):
        X, labels_true = data
        labels = np.unique(labels_true)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        colors = 'rgbycm'
        for i, label in enumerate(labels):
            position = labels_true == label
            ax.scatter(X[position, 0], X[position, 1], label="cluster %d" % label),
            color = colors[i % len(colors)]

        ax.legend(loc="best", framealpha=0.5)
        ax.set_xlabel("X[0]")
        ax.set_ylabel("Y[1]")
        ax.set_title("data")
        plt.show()

    # 测试函数
    def test_DBSCAN(*data):
        X, labels_true = data
        clst = cluster.DBSCAN();
        predict_labels = clst.fit_predict(X)
        print("ARI:%s" % adjusted_rand_score(labels_true, predict_labels))
        print("Core sample num:%d" % len(clst.core_sample_indices_))
    # #结果
    # ARI:0.330307120902
    # Core sample num:991
    #下面考察ϵ参数的影响：
    def test_DBSCAN_epsilon(*data):
        X, labels_true = data
        epsilons = np.logspace(-1, 1.5)
        ARIs = []
        Core_nums = []
        for epsilon in epsilons:
            clst = cluster.DBSCAN(eps=epsilon)
            predicted_labels = clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
            Core_nums.append(len(clst.core_sample_indices_))
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(epsilons, ARIs, marker='+')
        ax.set_xscale('log')
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylim(0, 1)
        ax.set_ylabel('ARI')
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(epsilons, Core_nums, marker='o')
        ax.set_xscale('log')
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel('Core_num')
        fig.suptitle("DBSCAN")
        plt.show()

    # centers = [[1,1],[1,2],[2,2],[10,20]]
    # X,labels_true = create_data(centers,1000,0.5)
    # test_DBSCAN_epsilon(X,labels_true)
    # 可以看到ARI指数随着ϵ的增长，先上升后保持平稳，最后悬崖式下降。悬崖式下降是因为我们产生的训练样本的间距比较小，最远的两个样本之间的距离不超过30，当ϵ过大时，所有的点都在一个邻域中。
    # 样本核心数量随着ϵ的增长而上升，这是因为随着ϵ的增长，样本点的邻域在扩展，则样本点邻域中的样本会增多，
    # 这就产生了更多满足条件的核心样本点。但是样本集中的样本数量有限，因此核心样本点的数量增长到一定数目后会趋于稳定。

    def test_DBSCAN_min_samples(*data):
        X,labels_true=data
        min_samples=range(1,100)
        ARIs=[]
        Core_nums=[]
        for num in min_samples:
            clst=cluster.DBSCAN(min_samples=num)
            predicted_labels=clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
            Core_nums.append(len(clst.core_sample_indices_))

        fig=plt.figure(figsize=(10,5))
        ax=fig.add_subplot(1,2,1)
        ax.plot(min_samples,ARIs,marker='+')
        ax.set_xlabel("min_samples")
        ax.set_ylim(0,1)
        ax.set_ylabel('ARI')

        ax=fig.add_subplot(1,2,2)
        ax.plot(min_samples,Core_nums,marker='o')
        ax.set_xlabel("min_samples")
        ax.set_ylabel('Core_nums')

        fig.suptitle("DBSCAN")
        plt.show()

    centers = [[1,1],[1,2],[2,2],[10,20]]
    X,labels_true = create_data(centers,1000,0.5)
    test_DBSCAN_min_samples(X,labels_true)








