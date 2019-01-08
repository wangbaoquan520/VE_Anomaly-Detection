# -*-coding:utf-8-*-
import math
from sklearn import metrics
from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans

# def NMI(A,B):
# # len(A) should be equal to len(B)
#     total = len(A)
#     A_ids = set(A)
#     B_ids = set(B)
#     #Mutual information
#     MI = 0
#     eps = 1.4e-45
#     for idA in A_ids:
#         for idB in B_ids:
#             idAOccur = np.where(A==idA)
#             idBOccur = np.where(B==idB)
#             idABOccur = np.intersect1d(idAOccur,idBOccur)
#             px = 1.0*len(idAOccur[0])/total
#             py = 1.0*len(idBOccur[0])/total
#             pxy = 1.0*len(idABOccur)/total
#             MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
#         # Normalized Mutual information
#         Hx = 0
#         for idA in A_ids:
#             idAOccurCount = 1.0*len(np.where(A==idA)[0])
#             Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
#         Hy = 0
#         for idB in B_ids:
#             idBOccurCount = 1.0*len(np.where(B==idB)[0])
#             Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
#         MIhat = 2.0*MI/(Hx+Hy)
#         return MIhat
# def tesrMNI():
#     A = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3])
#     B = np.array([1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3])
#     print (NMI(A,B))
#
# metrics
#
# """
#     聚类性能评估
# """
#
#
# """
#     2、Mutual Information based scores (MI) 互信息
#     优点：除取值范围在［0，1］之间，其他同ARI；可用于聚类模型选择
#     缺点：需要先验知识
#
# """
# labels_true = [0, 0, 0, 1, 1, 1];
# labels_pre5 = [0, 0, 1, 1, 2, 2];
# labels_pre6 = [1, 1, 0, 0, 3, 3];
# labels_pre7 = [1, 3, 2, 3, 2, 1];
# labels_pre8 = [1, 1, 1, 2, 2, 2]; # 完美聚类，聚类结果与初始类别完全一致，此时评估结果为：1.0
#
# print(metrics.adjusted_mutual_info_score(labels_pred=labels_pre5, labels_true=labels_true));
# print(metrics.adjusted_mutual_info_score(labels_pred=labels_pre6, labels_true=labels_true));
# print(metrics.adjusted_mutual_info_score(labels_true=labels_true, labels_pred=labels_pre7));
# print(metrics.adjusted_mutual_info_score(labels_true=labels_true, labels_pred=labels_pre8));
# print('-----------------------------------------------------------------------------------');
#
# """
#     3、Homogeneity, completeness and V-measure
#     同质性homogeneity：每个群集只包含单个类的成员。
#     完整性completeness：给定类的所有成员都分配给同一个群集。
#     两者的调和平均V-measure：
#     优点：[0，1]之间
# """
# labels_true = [0, 0, 0, 1, 1, 1];
# labels_pre9 = [0, 0, 1, 1, 2, 2];
# labels_pre10 = [0, 0, 0, 1, 1, 1];
#
# print(metrics.homogeneity_score(labels_true, labels_pre9));
# print(metrics.completeness_score(labels_pre9, labels_true));
# print(metrics.completeness_score(labels_pre10, labels_true));
# print(metrics.v_measure_score(labels_true, labels_pre9));
# print(metrics.homogeneity_completeness_v_measure(labels_true, labels_pre9));
#
# """
#     Example: 不同评分保准所依赖的计算方式不同
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from time import time
# from sklearn import metrics
#
#
# def uniform_labelings_scores(score_func, n_samples, n_clusters_range, fixed_n_classes=None, n_runs=5, seed=42):
#     random_labels = np.random.RandomState(seed).randint
#     scores = np.zeros((len(n_clusters_range), n_runs))
#
#     if fixed_n_classes is not None:
#         labels_a = random_labels(low=0, high=fixed_n_classes, size=n_samples)
#
#     for i, k in enumerate(n_clusters_range):
#         for j in range(n_runs):
#             if fixed_n_classes is None:
#                 labels_a = random_labels(low=0, high=k, size=n_samples)
#             labels_b = random_labels(low=0, high=k, size=n_samples)
#             scores[i, j] = score_func(labels_a, labels_b)
#     return scores
#
#
# score_funcs = [
#     metrics.adjusted_rand_score,
#     metrics.v_measure_score,
#     metrics.adjusted_mutual_info_score,
#     metrics.mutual_info_score,
# ]
#
# n_samples = 100
# n_clusters_range = np.linspace(2, n_samples, 10).astype(np.int)
#
# plt.figure(1)
#
# plots = []
# names = []
# for score_func in score_funcs:
#     print("Computing %s for %d values of n_clusters and n_samples=%d"
#           % (score_func.__name__, len(n_clusters_range), n_samples))
#
#     t0 = time()
#     scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range)
#     print("done in %0.3fs" % (time() - t0))
#     plots.append(plt.errorbar(
#         n_clusters_range, np.median(scores, axis=1), scores.std(axis=1))[0])
#     names.append(score_func.__name__)
#
# plt.title("Clustering measures for 2 random uniform labelings\n"
#           "with equal number of clusters")
# plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
# plt.ylabel('Score value')
# plt.legend(plots, names)
# plt.ylim(ymin=-0.05, ymax=1.05)
#
# n_samples = 1000
# n_clusters_range = np.linspace(2, 100, 10).astype(np.int)
# n_classes = 10
#
# plt.figure(2)
#
# plots = []
# names = []
# for score_func in score_funcs:
#     print("Computing %s for %d values of n_clusters and n_samples=%d"
#           % (score_func.__name__, len(n_clusters_range), n_samples))
#
#     t0 = time()
#     scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range,
#                                       fixed_n_classes=n_classes)
#     print("done in %0.3fs" % (time() - t0))
#     plots.append(plt.errorbar(
#         n_clusters_range, scores.mean(axis=1), scores.std(axis=1))[0])
#     names.append(score_func.__name__)
#
# plt.title("Clustering measures for random uniform labeling\n"
#           "against reference assignment with %d classes" % n_classes)
# plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
# plt.ylabel('Score value')
# plt.ylim(ymin=-0.05, ymax=1.05)
# plt.legend(plots, names)
# plt.show()
# print('-----------------------------------------------------------------------------')



# -----------------------------------  无监督  ------------------------------------------
"""
6、Calinski-Harabaz Index
优点：
6.1 评分很高时，簇的密度越高，划分越好，这也关系到一个聚类的标准性
6.2 评分是计算速度
"""
# datasets = datasets.load_iris()
# X = datasets.data
# y = datasets.target
#
# kmeans_model = KMeans(n_clusters=2, random_state=1).fit(X)
# labels = kmeans_model.labels_
# print(metrics.calinski_harabaz_score(X, labels))

def calinski_harabaz_score(X,labels):
    return metrics.calinski_harabaz_score(X, labels)

"""
5、 Silhouette Coefficient-轮廓系数
优点：
5.1 评分结果在[-1, +1]之间，评分结果越高，聚类结果越好
5.2 评分很高时，簇的密度越高，划分越好，这也关系到一个聚类的标准性
5.3 重要的是，这个评分标准不需要先验知识
"""
# datasets = datasets.load_iris()
# X = datasets.data
# y = datasets.target
#
# import numpy as np
# from sklearn.cluster import KMeans
#
# kmeans_model = KMeans(n_clusters=2).fit(X)
# labels = kmeans_model.labels_
# print(metrics.silhouette_score(X, labels, metric='euclidean'))

"""
    Example: silhouette analysis on KMeans clustering
"""
# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_samples, silhouette_score
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np
#
# X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True,
#                   random_state=1)
#
# range_n_clusters = [2, 3, 4, 5, 6]
#
# for n_clusters in range_n_clusters:
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)
#
#     # 第一个图是silhouette plot
#     # silhouette plot的坐标是[-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
#
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)
#
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print('For n_clusters = %d,' % n_clusters, 'The average silhouette_score is : %f' % silhouette_avg)
#
#     # 计算silhouette scores
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)
#
#     y_lower = 10
#     for i in range(n_clusters):
#         # 计算不同类簇的silhouette scores，并且进行排序
#         ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
#         ith_cluster_silhouette_values.sort()
#
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
#
#         color = cm.spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)
#
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#         y_lower = y_upper + 10
#
#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")
#
#     # 在silhouette_avg值处画垂直线
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#     ax1.set_yticks([])
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#     # 第二个图显示真实的聚类形式
#     colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors)
#
#     # 聚类中心
#     centers = clusterer.cluster_centers_
#     # 在聚类中心画圈
#     ax2.scatter(centers[:, 0], centers[:, 1],
#                 marker='o', c="white", alpha=1, s=200)
#
#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)
#
#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
#
#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')
#     plt.show()
# print('-----------------------------------------------------------------------------')

def Silhouette_Coefficient_score(X,labels):
    return metrics.silhouette_score(X, labels, metric='euclidean')

def Davies_Bouldin_score(X,labels):
    return metrics.davies_bouldin_score(X,labels)


# -----------------------------------  有监督  ------------------------------------------
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# Also if you need to compute Inverse Purity, all you need to do is replace "axis=0" by "axis=1".

"""
4、 Fowlkes-Mallows scores(FMI)
优点：[0, 1]
"""
# labels_true = [0, 0, 0, 1, 1, 1]
# labels_pre11 = [0, 0, 1, 1, 2, 2]
# labels_pre12 = [0, 0, 0, 1, 1, 1]
# labels_pre13 = [0, 1, 2, 0, 3, 4]
#
# print(metrics.fowlkes_mallows_score(labels_true, labels_pre11));
# print(metrics.fowlkes_mallows_score(labels_true, labels_pre12));
# print(metrics.fowlkes_mallows_score(labels_true, labels_pre13));

def FMI(labels_true,labels_pre):
    return metrics.fowlkes_mallows_score(labels_true, labels_pre)

"""
1、Adjusted Rand index (ARI)
优点：
1.1 对任意数量的聚类中心和样本数，随机聚类的ARI都非常接近于0；
1.2 取值在［－1，1］之间，负数代表结果不好，越接近于1越好；
1.3 可用于聚类算法之间的比较
缺点：
1.4 ARI需要真实标签
"""
# labels_true = [0, 0, 0, 1, 1, 1]
# labels_pre1 = [0, 0, 1, 1, 2, 2]
# labels_pre2 = [1, 1, 2, 2, 3, 3]
# labels_pre3 = [1, 1, 2, 2, 2, 1]
# labels_pre4 = [1, 1, 1, 2, 2, 2]  # 完美聚类，聚类结果与初始类别完全一致，此时评估结果为：1.0
#
# print(metrics.adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pre1))
# print(metrics.adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pre2))
# print(metrics.adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pre3))
# print(metrics.adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pre4))
# print('--------------------------------------------------------------------------------')
def ARI(labels_true,labels_pre):
    return metrics.adjusted_rand_score(labels_true, labels_pre)

def NMI(labels_true,labels_pre):
    return metrics.normalized_mutual_info_score(labels_true, labels_pre)

def Jaccard_Score(labels_true,labels_pre):
    return metrics.jaccard_similarity_score(labels_true,labels_pre)

def MSE(labels_true,labels_pre):
    return metrics.mean_squared_error(labels_true,labels_pre)

def Accuracy(labels_true,labels_pre):
    return metrics.accuracy_score(labels_true,labels_pre)

def Fmeasure(labels_true,labels_pre):
    return metrics.f1_score(labels_true,labels_pre)

def AUC():
    return metrics.auc()
