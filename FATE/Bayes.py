def priorProbability(labelList):  # 计算先验概率
    labelSet = set(labelList)  # 得到类别的值
    labelCountDict = {}   # 利用一个字典来存储训练集中各个类别的实例数
    for label in labelList:
        if label not in labelCountDict:
            labelCountDict[label] = 0
        labelCountDict[label] += 1
    priorProbabilityDict = {}
    for label in labelSet:  # 计算不同的类别对应的先验概率
        priorProbabilityDict[label] = labelCountDict[label]/len(labelList)
    return priorProbabilityDict

def conditionProbability(dataSet,labelList):  # 计算条件概率
    dimNum = len(dataSet[0]) # 得到特征数
    characterVal = []
    # 利用一个数组来存储训练数据集中不同特征的不同特征值。
    # 每一个不同特征的特征值都要需要另一个数组来存储，这样 characterVal实际上是一个二维数组
    for i in range(dimNum):
        temp = []
        for j in range(len(dataSet)):
            if dataSet[j][i] not in temp:
                temp.append(dataSet[j][i])
        characterVal.append(temp)
    probability = []  # 数组来存储最后的所有的条件概率
    labelSet = list(set(labelList))
    for dim in range(dimNum):  # 学习条件概率，需要计算K*S1*...*Sj个概率
        tempMemories = {}  # 对于每一个特征，利用一个字点来存储这个特征所有的取值对应的条件概率
        for val in characterVal[dim]:
            for label in labelSet:
                labelCount = 0  # 记录每一类的个数
                mixCount = 0  # 记录当前特征值为这个数，且类别为这个类别的实例个数
                for i in range(len(labelList)):
                    if labelList[i] == label:
                        labelCount += 1
                        if dataSet[i][dim] == val:
                            mixCount += 1
                tempMemories[str(val) + "|" + str(label)] = mixCount/labelCount
                # key表示哪一个特征值和类别，键表示对应的条件概率
        probability.append(tempMemories)  # 计算完一个特征，填充一个
    return probability  # 返回条件概率

def naiveBayes(x,dataSet,labelList): # 贝叶斯分类

    priorProbabilityDict = priorProbability(labelList)
    probability = conditionProbability(dataSet,labelList)
    bayesProbability = {}  # 计算所有类所对应的后验概率
    labelSet = list(set(labelList))
    for label in labelSet:
        tempProb = priorProbabilityDict[label]
        for dim in range(len(x)):
            tempProb *= probability[dim][str(x[dim])+"|"+str(label)]
        bayesProbability[label] = tempProb
    result = sorted(bayesProbability.items(),key= lambda x:x[1],reverse=True)# 排序
    return result[0][0]  # 返回后验概率最大的类

dataSet = ([[1,"s"],[1,"m"],[1,"m"],[1,"s"],[1,"s"],[2,"s"],[2,"m"],[2,"m"]])
labelList = [-1,-1,1,1,-1,-1,-1,1]
print(naiveBayes([2,"s"],dataSet,labelList))
## 返回结果为-1，即归为-1类。
