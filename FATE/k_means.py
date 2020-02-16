from numpy import *
import matplotlib.pyplot as plt
import rayleigh
# import numpy

# 加载本地数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # print(curLine)
        fltLine = list(map(float, curLine))  # 映射所有数据为浮点数
        # print(fltLine)
        dataMat.append(fltLine)
    return dataMat

# 欧式距离计算
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 格式相同的两个向量做运算

# 中心点生成 随机生成最小到最大值之间的值
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # 创建中心点，由于需要与数据向量做运算，所以每个中心点与数据得格式应该一致（特征列）
    for j in range(n):  # 循环所有特征列，获得每个中心点该列的随机值
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))  # 获得每列的随机值 一列一列生成
    return centroids

# 返回 中心点矩阵和聚类信息
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # 创建一个矩阵用于记录该样本 （所属中心点 与该点距离）
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False  # 如果没有点更新则为退出

        for i in range(m):
            # inf 正无穷  non 负无穷
            minDist = inf;
            minIndex = -1
            for j in range(k):  # 每个样本点需要与 所有 的中心点作比较
                distJI = distMeas(centroids[j, :], dataSet[i, :])  # 距离计算
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 若记录矩阵的第i个样本的所属中心点更新，则为True，while下次继续循环更新
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2  # 记录该点的两个信息
        # print(centroids)

        for cent in range(k):  # 重新计算中心点
            # nonzero[0]返回True样本的下标    nonzero[1]返回False样本的下标     得到属于该中心点的所有样本数据
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # axis=0 列求均
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# 二分均值聚类    SSE误差平方和    nonzero判断非0或给定条件，返回两个数组，[0]为True的下标组
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]

    clusterAssment = mat(zeros((m, 2)))  # 保存数据点的信息（所属类、误差）

    # tolist() 将数组或者矩阵变为列表
    # print(mean(dataSet, axis=0).tolist())
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 根据数据集均值获得第一个簇中心点


    centList = [centroid0]  # 创建一个带有质心的 [列表]，因为后面还会添加至k个质心
    # print(len(centList))
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2  # 求得dataSet点与质心点的SSE

    while (len(centList) < k):

        lowestSSE = inf

        for i in range(len(centList)):

            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]  # 与上面kmeans一样获得属于该质心点的所有样本数据

            # 二分类
            if len(ptsInCurrCluster):
                centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # 返回中心点信息、该数据集聚类信息
                # print('ok')
            else:
                splitClustAss = mat([1, 0])
                centroidMat = centroid0
                # print('empty')


            sseSplit = sum(splitClustAss[:, 1])  # 这是划分数据的SSE

            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])  # 这是未划分的数据集的SSE

            # print("划分SSE, and 未划分SSE: ", sseSplit, sseNotSplit)

            if (sseSplit + sseNotSplit) < lowestSSE:  # 将划分与未划分的SSE求和与最小SSE相比较 确定是否划分

                bestCentToSplit = i  # 得出当前最适合做划分的中心点

                bestNewCents = centroidMat  # 划分后的两个新中心点

                bestClustAss = splitClustAss.copy()  # 划分点的聚类信息

                lowestSSE = sseSplit + sseNotSplit

        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # 由于是二分，所以只有0，1两个簇编号，将属于1的所属信息转为下一个中心点

        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  # 将属于0的所属信息替换用来聚类的中心点

        # print('本次最适合划分的质心点: ', bestCentToSplit)

        # print('被划分数据数量: ', len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # 与上面两条替换信息相类似，这里是替换中心点信息，上面是替换数据点所属信息

        centList.append(bestNewCents[1, :].tolist()[0])   # 添加中心点

        # bestCentToSplit 代替被划分中心点的label
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  # 替换部分用来聚类的数据的所属中心点与误差平方和为新的数据

    return mat(centList), clusterAssment


def k_means_run(clust_num):
    datMat = mat(loadDataSet('testSet.txt'))
    # print(datMat)
    myCentroids, clustAssing = kMeans(datMat, clust_num)
    # print(myCentroids)
    # print(clustAssing)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], color='r', s=60)
    ax.scatter(datMat[:, 0].flatten().A[0], datMat[:, 1].flatten().A[0])
    # plt.show()

def b_k_meams_run(clust_num,datMat_temp):

    # datMat3 = mat(loadDataSet('testSet.txt'))

    datMat_temp = mat(datMat_temp)
    # print(datMat_temp)
    centList, myNewAssments = biKmeans(datMat_temp, clust_num)

    # print(centList)
    # print(myNewAssments[:,0][1])


    # 画图 注释处3   共3处
    # ax = fig.add_subplot(111)
    # ax.scatter(centList[:, 0].flatten().A[0], centList[:, 1].flatten().A[0], color='r', s=300, marker='3')
    # ax.scatter(datMat_temp[:, 0].flatten().A[0], datMat_temp[:, 1].flatten().A[0])


    return myNewAssments[:,0], centList



step_final = random.randint(0,100,[20,2])
# 画图 注释处1   共3处
# fig = plt.figure()
# plt.ion()
# plt.show()

cluster_membernum = 20
cluster_num = 3
for i in range(10):
    step_len = random.normal(0,2,[cluster_membernum, 2])
    step_final = step_final + step_len

    #此处添加范围限制
    for border_num in range(20):
        if step_final[border_num][0] >= 100:
            step_final[border_num][0] = 100
        if step_final[border_num][0] <= 0:
            step_final[border_num][0] = 0
        if step_final[border_num][1] >= 100:
            step_final[border_num][1] = 100
        if step_final[border_num][1] <= 0:
            step_final[border_num][1] = 0



    myNewAssments_temp, centList_temp = b_k_meams_run(cluster_num, step_final)
    # print(step_final)
    # print(myNewAssments_temp)
    # print(centList_temp)

    # print(type(centList_temp))
    cluster_array = []
    threshold_distribute = []
    common_point_num = []
    centList_temp_location = []

    # 将每一簇的数据分开
    for k in range(cluster_num):
        cluster_temp = [i for i, d in enumerate(myNewAssments_temp) if d == k]

        # 单独处理只有n-1个中心点的情况  n为所分簇的个数
        if len(cluster_temp):
            cluster_array.append(cluster_temp)
            common_point_num.append(1) # 对应P数目  P点放最后
            threshold_distribute.append(rayleigh.threshold90)
            centList_temp_location.append(k)
        else:
            delete(centList_temp, k, axis=1)
            # print(centList_temp)
            # 缺少中心点时加入空数组以保证索引值不变
            cluster_array.append([])
            # print('empty')
            continue

    # 计算各个移动端到达中心点的距离
    # print(array(centList_temp_location))
    # print(cluster_array)
    dist_cluster_array = []
    dist_array = []
    for p in array(centList_temp_location):
        dist_cluster_array = []
        for q in cluster_array[p]:
            # print(cluster_array[p])
            dist_cluster_array.append(distEclud(step_final[q], centList_temp[p]))
            # print(dist_cluster_array)
        dist_array.append(dist_cluster_array)
    # print(dist_array)


    # print(cluster_array)
    # print(common_point)
    # print(threshold_distribute)

    # 画图 注释处2   共两处
    # plt.pause(0.5)
    # plt.clf()





