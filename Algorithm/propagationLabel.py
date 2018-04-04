# 功能：跟标签有关的所有函数
# 编写人：马博渊，时间：2017-2-24

import numpy as np
from skimage.measure import label


def reviseLabel(labeled):
    tempLabeled = labeled.copy()
    for indexX in range(0, labeled.shape[0]):
        for indexY in range(0, labeled.shape[1]):
            if labeled[indexX,indexY] == 0:
                if indexX != 0 and tempLabeled[indexX-1,indexY] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX-1,indexY]
                    continue
                elif indexY != 0 and tempLabeled[indexX,indexY-1] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX,indexY-1]
                    continue
                elif indexX != labeled.shape[0]-1 and tempLabeled[indexX+1,indexY] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX+1,indexY]
                    continue
                elif indexY != labeled.shape[1]-1 and tempLabeled[indexX,indexY+1] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX,indexY+1]
                    continue
                elif indexX != 0 and indexY != 0 and tempLabeled[indexX-1,indexY-1] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX-1,indexY-1]
                    continue
                elif indexX != 0 and indexY != labeled.shape[1]-1 and tempLabeled[indexX-1,indexY+1] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX-1,indexY+1]
                    continue
                elif indexX != labeled.shape[0]-1 and indexY != 0 and tempLabeled[indexX+1,indexY-1] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX+1,indexY-1]
                    continue
                elif indexX != labeled.shape[0]-1 and indexY != labeled.shape[1]-1 and tempLabeled[indexX+1,indexY+1] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX+1,indexY+1]
                    continue
                elif indexX >= 1 and tempLabeled[indexX-2,indexY] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX-2,indexY]
                    continue
                elif indexY >= 1 and tempLabeled[indexX,indexY-2] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX,indexY-2]
                    continue
                elif indexX <= labeled.shape[0]-2 and tempLabeled[indexX+2,indexY] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX+2,indexY]
                    continue
                elif indexY <= labeled.shape[1]-2 and tempLabeled[indexX,indexY+2] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX,indexY+2]
                    continue
                elif indexX >= 2 and tempLabeled[indexX-3,indexY] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX-3,indexY]
                    continue
                elif indexY >= 2 and tempLabeled[indexX,indexY-3] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX,indexY-2]
                    continue
                elif indexX <= labeled.shape[0]-2 and tempLabeled[indexX+2,indexY] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX+2,indexY]
                    continue
                elif indexY <= labeled.shape[1]-2 and tempLabeled[indexX,indexY+2] != 0:
                    labeled[indexX,indexY] = tempLabeled[indexX,indexY+2]
                    continue
    return labeled

def normalizeLabel(labeled):
    """
    功能：归一化Label
    """
    maxNum = np.max(labeled)
    tempIndex = 0
    for indexNum in range(0, maxNum+1):
        for index in range(indexNum+1, maxNum+1):
            if index not in labeled:
                continue
            else:
                labeled[labeled == index] = indexNum+1
                break
    return labeled

def denoiseByArea(grayImage, areaThresh, neighbors = 8):
    """
    功能： 去掉面积小于一定阈值的噪声点
        输入： grayImage:灰度图像
              areaThresh: 面积阈值
              neighbors:4邻域或8邻域（可选，默认4邻域）
        输出: 去掉噪声的图
    """
    labeled, num = label(grayImage, neighbors=neighbors, return_num=True)
    zerosMatrix = np.zeros(labeled.shape)

    for index in range(1, num+1):
        zerosMatrix[labeled == index] = 1
        temp = np.count_nonzero(zerosMatrix)
        if temp < areaThresh:
            grayImage[labeled == index] = 0
        zerosMatrix = np.zeros(labeled.shape)
    return grayImage

def addAllComparation(segLabeled, truLabeled, boundingImage, binaryCostMatrix):
    """
    功能：输出全对比图像,传播后图像边缘为绿色，GroundTruth为红色，boundingRegion为紫色
        输出：结果RGB图像
        :param segLabeled:
    """
    returnImage = np.zeros((segLabeled.shape[0], segLabeled.shape[1], 3))   # 建立需要输出图像
    segEdge = getEdgesFromLabel(segLabeled, neighbors=4)                    # 下一张分割后的边缘图
    truEdge = getEdgesFromLabel(truLabeled, neighbors=4)                    # 下一张人工标注的边缘图

    # returnImage = binaryCost.addComparation(returnImage, binaryCostMatrix, color=np.array([102, 255, 255])) # binaryCost为蓝色

    for indexX in range(0, segLabeled.shape[0]):
        for indexY in range(0, segLabeled.shape[1]):
            if segEdge[indexX, indexY] == 255:
                returnImage[indexX, indexY] = [0, 255, 0]                        # 分割后的结果图为绿色（BGR）
            elif truEdge[indexX, indexY] == 255 and segEdge[indexX, indexY] != 255:
                returnImage[indexX, indexY] = [0, 0, 255]                        # 分割的真实图为红色（BGR）
            elif (np.array(boundingImage[indexX, indexY]) == [255, 255, 255]).all() and segEdge[indexX, indexY] != 255:
                returnImage[indexX, indexY] = [238, 122, 233]   # boundingRegion为紫色

    return returnImage

def addComparation(originalImage, comparation, value):
    """
    功能：将comparation的结果加到OriginalImage上
        输入： originalImage: 需要增加对比的原图，必须为灰度图像
               comparation:对比图（shape大小要和originalImage相同）
               value: 增加对比图后的像素值
        输出： 增加后的对比图(RGB)
    """
    if comparation.ndim == 3:
        whiteValue = np.array([255, 255, 255])
    elif comparation.ndim == 2:
        whiteValue = 255
    returnImage = originalImage.copy()
    for indexX in range(0, returnImage.shape[0]):
        for indexY in range(0, returnImage.shape[1]):
            if (np.array(comparation[indexX, indexY]) == whiteValue).all() and returnImage[indexX, indexY] != 255:
                if returnImage.ndim == 3:
                    returnImage[indexX, indexY] = [255,255, 255]
                elif returnImage.ndim == 2:
                    returnImage[indexX, indexY] = value
    return returnImage

def addOneInLabel(labeled):
    """
    功能：将标签里每个标签增加1，适用于graphCut结果增加1
        输入： label: graphCut结果，m * n * 1
        输出： label所有标签增加1
    """
    maxNumber = np.max(labeled)
    result = np.zeros(labeled.shape)
    for indexNumber in range(0, maxNumber+1):
        if indexNumber in labeled:
            result[labeled == indexNumber] = indexNumber + 1
    return result.astype(np.int32)

def countSameArea(value, segementLabel, groundTruthLabel):
    """
    功能：计算某个标签值在分割结果和真实值中重合的面积
        输入： value: 需要计算的label
               segmentLabel: 分割后的标签图
               groundTruthLabel:分割的真实标签图
        输出： 该标签在两图中的重合面积
    """
    area = 0
    for indexX in range(0, groundTruthLabel.shape[0]):
        for indexY in range(0, groundTruthLabel.shape[1]):
            if segementLabel[indexX, indexY] == groundTruthLabel[indexX, indexY] == value:
                area += 1
    return area

def evaluateDice(segmentLabel, groundTruthLabel):
    """
    功能：Dice 评测，对比分割结果和真实值
        输入： segmentLabel: 分割后的标签图
               groundTruthLabel:分割的真实标签图
        算法： Dice = ∑ weighit * 2 * Asame / (Aseg + Aground)
               weight 是某个标签在ground中所占比例，Asame是某个label在seg和ground重合部分
               Aseg是某个label在seg中的面积，Aground是某个label在ground中的面积，
        输出： Dice评估值
    """
    totalPixel = segmentLabel.shape[0] * segmentLabel.shape[1]
    particleNumberInSeg = getNumberFromLabel(segmentLabel)
    particleNumberInTru = getNumberFromLabel(groundTruthLabel)
    maxNumberInSeg = np.max(segmentLabel)
    maxNumberInTru = np.max(groundTruthLabel)
    result = 0.0
    for indexNumber in range(1, maxNumberInTru+1):
        if indexNumber in segmentLabel and indexNumber in groundTruthLabel:
            weight = np.sum(groundTruthLabel == indexNumber) * 1.0 / totalPixel
            sameArea = countSameArea(indexNumber, segmentLabel, groundTruthLabel)
            result = result + weight * (2.0 * sameArea / (np.sum(groundTruthLabel == indexNumber) + np.sum(segmentLabel == indexNumber)))
    return result

def getDistanceFromGroundTruthPoint(groundTruthEdge, indexX, indexY, neighborLength = 60):
    """
    功能：计算检测到的边缘点与离它最近边缘点的距离
    """
    if groundTruthEdge[indexX,indexY] == 255:
        return 0


    distance = neighborLength/2
    findPoint = False
    regionStartRow = 0
    regionStartCol = 0
    regionEndRow = groundTruthEdge.shape[0]
    regionEndCol = groundTruthEdge.shape[1]
    if indexX - neighborLength > 0:
        regionStartRow = indexX - neighborLength
    if indexX + neighborLength < groundTruthEdge.shape[0]:
        regionEndRow = indexX + neighborLength
    if indexY - neighborLength > 0:
        regionStartCol = indexY - neighborLength
    if indexY + neighborLength < groundTruthEdge.shape[1]:
        regionEndCol = indexY + neighborLength

    indexXX = 0;indexYY=0;
    for indexGroundX in range(regionStartRow, regionEndRow):
        for indexGroundY in range(regionStartCol, regionEndCol):
            if groundTruthEdge[indexGroundX, indexGroundY] == 255:
                tempDisance = np.sqrt(pow(indexGroundX - indexX,2)+pow(indexGroundY - indexY,2))
                if tempDisance < distance:
                    distance = tempDisance
                    indexXX = indexGroundX
                    indexYY = indexGroundY
    # print("     indexXX="+str(indexXX)+",indexYY="+str(indexYY))
    return distance

def evaluateMeritForEdge(segmentEdge, groundTruthEdge, constIndex = 0.1):
    """
    功能：使用品质因素评价方法边缘检测质量

    """
    numSeg = np.sum(segmentEdge[segmentEdge == 255])/255
    numGround = np.sum(groundTruthEdge[groundTruthEdge == 255])/255
    # print("numSeg="+str(numSeg))
    # print("numGround="+str(numGround))
    maxNum = numSeg

    if numSeg < numGround:  maxNum = numGround
    # print("maxNum="+str(maxNum))
    temp = 0.0
    for indexX in range(0, segmentEdge.shape[0]):
        for indexY in range(0, segmentEdge.shape[1]):
             if segmentEdge[indexX, indexY] == 255:
                distance = getDistanceFromGroundTruthPoint(groundTruthEdge, indexX, indexY)
                # print(str(indexX)+", "+str(indexY)+":distance="+str(distance))
                temp = temp + 1 / (1 + constIndex * pow(distance,2))
    merit = (1.0 / maxNum) * temp
    return merit

# ******************************** 评估算法准确性 *****************************************
def evaluatePrecisonRecallForLabeld(segmentLabel, groundTruthLabel):
    """
    功能：Precision 和 Recall 评测，对比分割结果和真实值
        输入： segmentLabel: 分割后的标签图
               groundTruthLabel:分割的真实标签图
        算法： precision = ∑ weighit * Asame / Aseg; recall = ∑ weighit * Asame / Aground;
               weight 是某个标签在ground中所占比例，Asame是某个label在seg和ground重合部分
               Aseg是某个label在seg中的面积，Aground是某个label在ground中的面积，
        输出： Precision 和 Recall 评测
    """
    totalPixel = segmentLabel.shape[0] * segmentLabel.shape[1]
    particleNumberInSeg = getNumberFromLabel(segmentLabel)
    particleNumberInTru = getNumberFromLabel(groundTruthLabel)
    maxNumberInSeg = np.max(segmentLabel)
    maxNumberInTru = np.max(groundTruthLabel)
    precision = 0.0
    recall = 0.0
    for indexNumber in range(1, maxNumberInTru+1):
        if indexNumber in segmentLabel and indexNumber in groundTruthLabel:
            weight = np.sum(groundTruthLabel == indexNumber) * 1.0 / totalPixel
            sameArea = countSameArea(indexNumber, segmentLabel, groundTruthLabel)
            precision = precision + weight * (sameArea / np.sum(segmentLabel == indexNumber))
            recall = recall + weight * (sameArea / np.sum(groundTruthLabel == indexNumber))
    return (precision, recall)

def getEdgesFromLabel(labeled, neighbors=4):
    """
    功能：根据label获得边界图像
        输入： labeled: 分割后的标签图
               neighbor: 4邻域或8邻域，（可选，默认4邻域）
        算法： 某个像素的邻域内若有和自己标签不一样的像素，则该像素为边界像素，设为255，否则为0
        输出： edgeImage; (edgeImage == 0 or edgeImage == 255)
    """
    edges = np.zeros(labeled.shape)
    for row in range(0, labeled.shape[0]):
        for col in range(0, labeled.shape[1]):
            value = labeled[row, col]
            isEdges = False
            if neighbors == 4:
                if row != 0 and value != labeled[row-1, col]:
                    isEdges = True
                elif col != 0 and value != labeled[row, col-1]:
                    isEdges = True
                elif row != labeled.shape[0]-1 and value != labeled[row+1, col]:
                    isEdges = True
                elif col != labeled.shape[1]-1 and value != labeled[row, col+1]:
                    isEdges = True
            elif neighbors == 8:
                if row != 0 and value != labeled[row-1, col]:
                    isEdges = True
                elif row != 0 and col != 0 and value != labeled[row-1, col-1]:
                    isEdges = True
                elif row != 0 and col != labeled.shape[1]-1 and value != labeled[row-1, col+1]:
                    isEdges = True
                elif row != labeled.shape[0]-1 and value != labeled[row+1, col]:
                    isEdges = True
                elif row != labeled.shape[0]-1 and col != 0 and value != labeled[row+1, col-1]:
                    isEdges = True
                elif row != labeled.shape[0]-1 and col != labeled.shape[1]-1 and value != labeled[row+1, col+1]:
                    isEdges = True
                elif col != 0 and value != labeled[row, col-1]:
                    isEdges = True
                elif col != labeled.shape[1]-1 and value != labeled[row, col+1]:
                    isEdges = True
            if isEdges == True:
                edges[row, col] = 255

    return edges

def joinInEqualTabel(value1, value2, equalTable):
    """
    功能：把冲突的标号加入等价表
        输入： value1: 需要加入等价表中的数值1
               value2：需要加入等价表中的数值2
        算法： equalTable是个列表，列表中每个元素又是列表，存储等价关系
        输出： equalTable
    """
    if equalTable is None or len(equalTable) == 0:
        equalTable = []
        dList = [int(value1), int(value2)]
        dList.sort()
        equalTable.append(dList)
    for index in range(0, len(equalTable)):
        if value1 in equalTable[index] and value2 in equalTable[index]:
            return equalTable
        elif value1 in equalTable[index] and value2 not in equalTable[index]:
            equalTable[index].append(value2)
            equalTable[index].sort()
            return equalTable
        elif value2 in equalTable[index] and value1 not in equalTable[index]:
            equalTable[index].append(value1)
            equalTable[index].sort()
            return equalTable
    dList = [int(value1), int(value2)]
    dList.sort()
    equalTable.append(dList)
    return equalTable

def getNumberFromLabel(labeled):
    """
    功能：从label中得到粒子数
        输入： labeled: 标签图
        算法： 考虑到标签可能不连续，先unique再求length
        输出： 标签图中的粒子数目
    """
    num = len(np.unique(labeled))
    if 0 in labeled:
        num = num - 1
    return num

def getRGBLabel(originalImage, neighbors=4, return_num=False):
    """
    功能：对RGB图像进行图像标注，得到标注结果（鉴于skimage的label仅适用灰度图像，特此开发此函数）
    目前只支持4邻接，谁有本事简化或加8邻域你自己来
        输入： originalImage: 需要标注的RGB或灰度或二值图像
               value2：需要加入等价表中的数值2
               return_num：是否返回label的粒子数目（True或False）（可选，默认为False）
        算法： 图像标记算法
        输出： label标记图，num粒子数目（由return_num控制）
    """
    label = np.zeros((originalImage.shape[0], originalImage.shape[1])).astype(np.int32)
    maxLabel = 1
    equalTable = []
    for indexX in range(0, originalImage.shape[0]):
        for indexY in range(0, originalImage.shape[1]):
            if neighbors == 4:                               # 4领域
                if indexX == 0 and indexY == 0:
                    label[indexX, indexY] = maxLabel
                    maxLabel += 1
                elif indexX == 0 and indexY > 0:
                    if (np.array(originalImage[indexX, indexY]) == np.array(originalImage[indexX, indexY-1])).all():
                        label[indexX, indexY] = label[indexX, indexY-1]
                    else:
                        label[indexX, indexY] = maxLabel
                        maxLabel += 1
                elif indexX > 0 and indexY == 0:
                    if (np.array(originalImage[indexX, indexY]) == np.array(originalImage[indexX-1, indexY])).all():
                        label[indexX, indexY] = label[indexX-1, indexY]
                    else:
                        label[indexX, indexY] = maxLabel
                        maxLabel += 1
                elif indexX > 0 and indexY > 0:
                    # print("位置：x="+str(indexX)+" y="+str(indexY)+" 左="+str(originalImage[indexX-1, indexY]) + " 上="+str(originalImage[indexX, indexY-1])+" 自己="+str(originalImage[indexX, indexY]) + " 左标签="+str(label[indexX-1, indexY])+" 上标签="+str(label[indexX, indexY-1]))
                    if (np.array(originalImage[indexX-1, indexY]) == np.array(originalImage[indexX, indexY-1])).all() and (np.array(originalImage[indexX, indexY]) == np.array(originalImage[indexX, indexY-1])).all():
                        label[indexX, indexY] = label[indexX, indexY-1]
                        if label[indexX-1, indexY] != label[indexX, indexY-1]:
                            equalTable = joinInEqualTabel(label[indexX-1, indexY], label[indexX, indexY-1], equalTable)
                    elif (np.array(originalImage[indexX-1, indexY]) == np.array(originalImage[indexX, indexY-1])).all() and (np.array(originalImage[indexX, indexY]) != np.array(originalImage[indexX, indexY-1])).any():
                        label[indexX, indexY] = maxLabel
                        maxLabel += 1
                    elif (np.array(originalImage[indexX-1, indexY]) == np.array(originalImage[indexX, indexY])).all() and (np.array(originalImage[indexX, indexY-1]) != np.array(originalImage[indexX, indexY])).any():
                        label[indexX, indexY] = label[indexX-1, indexY]
                    elif (np.array(originalImage[indexX, indexY-1]) == np.array(originalImage[indexX, indexY])).all() and (np.array(originalImage[indexX-1, indexY]) != np.array(originalImage[indexX, indexY])).any():
                        label[indexX, indexY] = label[indexX, indexY-1]
                    elif (np.array(originalImage[indexX, indexY-1]) != np.array(originalImage[indexX, indexY])).any() and (np.array(originalImage[indexX-1, indexY]) != np.array(originalImage[indexX, indexY])).any() and (np.array(originalImage[indexX-1, indexY]) != np.array(originalImage[indexX, indexY-1])).any():
                        label[indexX, indexY] = maxLabel
                        maxLabel += 1
    if equalTable is None or len(equalTable) == 0:
        if return_num == True:
            return (label, 0)
        elif return_num == False:
            return label
    # 归一化表，防止同样数据出现在等价表中的不同表项
    for indexX2 in range(0, len(equalTable)-1):
        for indexY2 in range(1, len(equalTable[indexX2])):
            for tempIndex in range(indexX2+1, len(equalTable)):
                if equalTable[indexX2][indexY2] in equalTable[tempIndex]:
                    tempList = equalTable.pop(tempIndex)
                    equalTable.append([])
                    for elementIndex in tempList:
                        equalTable[indexX2].append(elementIndex)
                    # break
    for element in equalTable:
        if element is None or len(element) == 0:
            equalTable.remove(element)
    # 根据等价表第二遍扫描图像
    for indexX1 in range(0, len(equalTable)):
        for indexY1 in range(1, len(equalTable[indexX1])):
            label[label == equalTable[indexX1][indexY1]] = equalTable[indexX1][0]
    for indexNum in range(1, np.max(label)):
        if indexNum+1 not in label:
            for indexA in range(indexNum+2, np.max(label)+1):
                if indexA in label:
                    label[label == indexA] = indexNum+1
                    break
    # print(np.max(label))
    if return_num == False:
        return label
    elif return_num == True:
        return (label, getNumberFromLabel(label))

def getLabelFromNighbor(startX, startY, nextHumanLabel, nextTempLabel, neighborLength=9):
    """
    功能：作为getNextImageLabel函数的辅助，根据邻域的像素得到此位置的标注
        输入： startX, startY: 需标注的像素在nextTempLabel的位置
               nextHumanLabel：下一张人为标记的label
               nextTempLabel：下一张标签矩阵
               eighborLength：邻域的边界长度，必须为单数（可选，默认为9）
        输出： 此位置的标注
    """
    if neighborLength < 3 or neighborLength % 2 == 0:
        return 1.0
    regionStartX = int(startX - (neighborLength-1)/2)
    if regionStartX < 0:
        regionStartX = 0
    regionStartY = int(startY - (neighborLength-1)/2)
    if regionStartY < 0:
        regionStartY = 0
    regionEndX = int(startX + (neighborLength-1)/2)
    if regionEndX > nextTempLabel.shape[0]:
        regionEndX = nextTempLabel.shape[0]
    regionEndY = int(startY + (neighborLength-1)/2)
    if regionEndY > nextTempLabel.shape[1]:
        regionEndY = nextTempLabel.shape[1]

    #print("startX="+str(startX)+" startY="+str(startY)+" regionStartX="+str(regionStartX)+" regionStartY="+str(regionStartY)+" regionEndX="+str(regionEndX)+" regionEndY="+str(regionEndY))
    value = 0
    for indexX in range(regionStartX, regionEndX):
        for indexY in range(regionStartY, regionEndY):
            if nextTempLabel[indexX][indexY] != 0 and (np.array(nextHumanLabel[indexX, indexY]) == np.array(nextHumanLabel[startX, startY])).all():
                value = nextTempLabel[indexX][indexY]
    return value

def getNextImageLabel(lastHumanLabelRGB, lastLabel, nextHumanLabelRGB, neighborLength=17, return_num=False):
    """
    功能：根据上一张人为标注的截面图像标号得到下一张人为标注的截面图像标号
          由于直接标注下一张人为标注图像容易出现同一粒子不同标号情况，所以专门开发此函数
          为提取groundTruth的标签和seg对比提供基础
        输入： lastHumanLabelRGB: 上一张人为标注的RGB图像
               lastLabel：上一张人为标记的label
               nextHumanLabelRGB：下一张人为标注的RGB图像
               neighborLength：领域的边界长度，必须为单数（可选，默认为17）
               return_num：是否返回label的粒子数目（True或False）（可选，默认为False）
        算法： 如果上一张人为标注图像和下一张人为标注图像某个同样位置的像素的像素值相同，则把上一张此像素的label赋给下一张；
               不符合上述条件的像素，则搜索邻域内的像素，如果有像素符合上述条件并且和自己像素相同，则将其标签赋给他
        输出： label标记图，num粒子数目（由return_num控制）
    """
    lastMax = np.max(lastLabel)
    # print("上一层的粒子数="+str(lastMax))
    nextTempLabel = np.zeros((nextHumanLabelRGB.shape[0], nextHumanLabelRGB.shape[1]))

    # 如果上一张和下一张的颜色一样，则把上一张的label赋值给下一张
    for indexX1 in range(0, nextHumanLabelRGB.shape[0]):
        for indexY1 in range(0, nextHumanLabelRGB.shape[1]):
            if (np.array(lastHumanLabelRGB[indexX1, indexY1]) == np.array(nextHumanLabelRGB[indexX1, indexY1])).all():
                nextTempLabel[indexX1, indexY1] = lastLabel[indexX1, indexY1]

    for indexX in range(0, nextHumanLabelRGB.shape[0]):
         for indexY in range(0, nextHumanLabelRGB.shape[1]):
            if nextTempLabel[indexX][indexY] == 0:
                nextTempLabel[indexX][indexY] = getLabelFromNighbor(indexX, indexY, nextHumanLabelRGB, nextTempLabel, neighborLength=neighborLength)
                if nextTempLabel[indexX][indexY] == 0:
                    nextTempLabel[indexX][indexY] = lastMax+1
                    # print("增加的粒子="+str(lastMax))
                    lastMax = lastMax+1
    if return_num == False:
        return nextTempLabel.astype(np.int32)
    elif return_num == True:
        return nextTempLabel.astype(np.int32), getNumberFromLabel(nextTempLabel)

