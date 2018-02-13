# 功能：计算binaryCost
# 方法：
#       1. 提供getBinaryCost函数
#       2. 提供gFunc函数
# 编写人：马博渊，时间：2017-2-24
import numpy as np
import cv2 as cv
from math import exp, pow
from .propagationLabel import denoiseByArea
from skimage import morphology
from skimage.future import graph
from skimage.measure import label
import matplotlib.pyplot as plt
from .unaryCost import getUnaryCost

def getIndexInUnaryCost(lastNum, unaryCostMatrix, rowIndex, colIndex):
    for index in range(0, lastNum):
        if unaryCostMatrix[rowIndex, colIndex, index] == 0:
            return index


def countZeroNumInUnaryCost(lastNum, unaryCostMatrix, rowIndex, colIndex):
    zeroNum = 0
    for index in range(0, lastNum):
        if unaryCostMatrix[rowIndex, colIndex, index] == 0:
            zeroNum = zeroNum + 1
    return zeroNum


def getBinaryCostFromUnary(nextOriginalImage, lastOrigianlImage, lastLabeled, lastNum, unaryCostMatrix, type ="intensityImage", edgeOriginalImage = np.zeros((1,1)) , neighborLength = 9, infiniteCost = 100, KCost = 3):
    """
    功能：获得本层待分割图片的Binary Cost
        输入：nextOriginalImage：上一张原始图像
              lasthumanLabeledRGB：人为标注的RGB图像
              lastLabeled:上一张的标记结果
              lastNum：上一层标记结果的label个数
              unaryCostMatrix：一元项能量值矩阵
              type：处理图像类型（"intensityImage"或"edgeImage"）（可选，默认为"intensityImage"）
              edgeOriginalImage：边界图像，如果type == "edgeImage",此项才有用
              neighborLength: 邻域边界长度，（可选，必须为单数，默认为9）
              infiniteCost：无穷大设定值，默认100
              KCost：二元项K值取值，默认为3
        输出: binaryCost，m * 3(m 是边的数目，第一列边起点，第二列是边终点，第三列是边权值)
    """
    # 对图像进行转换，如果是彩色图像则转换为灰度图像
    inputImageGray = np.zeros(nextOriginalImage.shape)
    if nextOriginalImage.ndim == 3:
        inputImageGray = cv.cvtColor(nextOriginalImage, cv.COLOR_BGR2GRAY)
    elif nextOriginalImage.ndim == 2:
        inputImageGray = nextOriginalImage

    # 初始化边（ m*3 ），每一行对应一条边，前两个值对于边的起点和终点，最后一个对应边的权值
    inds = np.arange(inputImageGray.size).reshape(inputImageGray.shape)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.zeros((horz.shape[0]+vert.shape[0], 3))
    edges[:, :-1] = np.vstack([horz, vert])    # np.vstack 矩阵的合并
    edges = edges.astype(np.int32)

    # 构建区域邻接图
    rag = graph.rag_mean_color(lastOrigianlImage, lastLabeled)
    # lastSegment = propagationLabel.getEdgesFromLabel(lastLabeled)

    # 将boundingRegion以外的区域权值设高
    for indexEdge in range(0, edges.shape[0]):
        startPoint = edges[indexEdge, 0].astype(np.int32)
        endPoint = edges[indexEdge, 1].astype(np.int32)
        startRowIndex, startColIndex = pointIndexToCoordinate(inputImageGray.shape[0], inputImageGray.shape[1], startPoint)
        endRowIndex, endColIndex = pointIndexToCoordinate(inputImageGray.shape[0], inputImageGray.shape[1], endPoint)
        zeroNumInStartPoint = countZeroNumInUnaryCost(lastNum, unaryCostMatrix, startRowIndex, startColIndex)
        zeroNumInEndPoint = countZeroNumInUnaryCost(lastNum, unaryCostMatrix, endRowIndex, endColIndex)
        if zeroNumInStartPoint > 1 or zeroNumInEndPoint > 1:
            continue
        indexStart = getIndexInUnaryCost(lastNum, unaryCostMatrix, startRowIndex, startColIndex)
        indexEnd = getIndexInUnaryCost(lastNum, unaryCostMatrix, endRowIndex, endColIndex)
        if indexStart == indexEnd:
            edges[indexEdge, 2] = infiniteCost

    # 结合下一层的图像将boundingRegion区域权值根据情况设置低
    indexEdge = 0
    for indexEdge in range(0, edges.shape[0]):
        if edges[indexEdge, 2] == 0:
            startPoint = edges[indexEdge, 0].astype(np.int32)
            endPoint = edges[indexEdge, 1].astype(np.int32)
            startRowIndex, startColIndex = pointIndexToCoordinate(inputImageGray.shape[0], inputImageGray.shape[1], startPoint)
            endRowIndex, endColIndex = pointIndexToCoordinate(inputImageGray.shape[0], inputImageGray.shape[1], endPoint)
            if type == "intensityImage":
                edges[indexEdge, 2] = gFunc(inputImageGray[startRowIndex, startColIndex], inputImageGray[endRowIndex, endColIndex], imageType=type, m=10)
            elif type == "edgeImage":
                edges[indexEdge, 2] = gFunc(edgeOriginalImage[startRowIndex, startColIndex], edgeOriginalImage[endRowIndex, endColIndex], imageType=type, m=10)

    # 结合上一层的图像将boundingRegion区域权值权值设置低, 设为K，默认为3
    indexEdge = 0
    for indexEdge in range(0, edges.shape[0]):
        if edges[indexEdge, 2] > 0 and edges[indexEdge, 2] <= infiniteCost:
            startPoint = edges[indexEdge, 0].astype(np.int32)
            endPoint = edges[indexEdge, 1].astype(np.int32)
            startRowIndex, startColIndex = pointIndexToCoordinate(inputImageGray.shape[0], inputImageGray.shape[1], startPoint)
            endRowIndex, endColIndex = pointIndexToCoordinate(inputImageGray.shape[0], inputImageGray.shape[1], endPoint)
            firstLabel = lastLabeled[startRowIndex, startColIndex]
            secondLabel = lastLabeled[endRowIndex, endColIndex]
            if firstLabel != secondLabel and secondLabel in rag.neighbors(firstLabel):  # 如果上一层两点处label不同，而且这两个label相邻，则要计算:
                edges[indexEdge, 2] = KCost

    return edges.astype(np.int32)


def addComparationByWaggoner(nextOriginalImage, binaryCost):

    returnLabel = np.zeros((nextOriginalImage.shape[0], nextOriginalImage.shape[1]))
    for index in range(0, binaryCost.shape[0]):
        if binaryCost[index , 2] > 0:
            rowIndex, colIndex = pointIndexToCoordinate(nextOriginalImage.shape[0], nextOriginalImage.shape[1], binaryCost[index, 0])
            returnLabel[rowIndex, colIndex] = 255
            rowIndex, colIndex = pointIndexToCoordinate(nextOriginalImage.shape[0], nextOriginalImage.shape[1], binaryCost[index, 1])
            returnLabel[rowIndex, colIndex] = 255
    return returnLabel


def addComparation(nextOriginalImage, binaryCost):

    returnLabel = np.ones((nextOriginalImage.shape[0], nextOriginalImage.shape[1])) * 1000
    for index in range(0, binaryCost.shape[0]):
        rowIndex, colIndex = pointIndexToCoordinate(nextOriginalImage.shape[0], nextOriginalImage.shape[1], binaryCost[index, 0])
        # print("***************************")
        # print("起点="+str(rowIndex) + "  终点="+str(colIndex))
        if returnLabel[rowIndex, colIndex] > binaryCost[index , 2]:
            returnLabel[rowIndex, colIndex] = binaryCost[index , 2]

        rowIndex, colIndex = pointIndexToCoordinate(nextOriginalImage.shape[0], nextOriginalImage.shape[1], binaryCost[index, 1])
        # print("起点="+str(rowIndex) + "  终点="+str(colIndex))
        if returnLabel[rowIndex, colIndex] > binaryCost[index , 2]:
            returnLabel[rowIndex, colIndex] = binaryCost[index , 2]
    return returnLabel

# 计算方差
def getVariance(startRowIndex, startColIndex, inputImageGray, neighborLength = 7):
    if neighborLength < 3 or neighborLength % 2 == 0:
        return 1.0
    regionStartX = int(startRowIndex - (neighborLength-1)/2)
    if regionStartX < 0:
        regionStartX = 0
    regionStartY = int(startColIndex - (neighborLength-1)/2)
    if regionStartY < 0:
        regionStartY = 0
    regionEndX = int(startRowIndex + (neighborLength-1)/2)
    if regionEndX > inputImageGray.shape[0] - 1:
        regionEndX = inputImageGray.shape[0] - 1
    regionEndY = int(startColIndex + (neighborLength-1)/2)
    if regionEndY > inputImageGray.shape[1] - 1:
        regionEndY = inputImageGray.shape[1] - 1

    num = (regionEndY - regionStartY) * (regionEndX - regionStartX + 1) + (regionEndY - regionStartY + 1) * (regionEndX - regionStartX)
    edgeValues = np.zeros((num, 1), dtype=np.float32)
    # print("regionStartX="+str(regionStartX)+" regionStartY="+str(regionStartY)+" regionEndX="+str(regionEndX)+" regionEndY="+str(regionEndY))
    index = np.int32(0)
    for indexX in range(regionStartX, regionEndX + 1):
        for indexY in range(regionStartY, regionEndY):
            value1 = float(inputImageGray[indexX, indexY])
            value2 = float(inputImageGray[indexX, indexY + 1])
            temp = float(pow((value1 - value2), 2))
            edgeValues[index, 0] = temp
            index = index + 1
            # print(str(index)+" : "+str(temp))
            # print("indexX="+str(indexX)+" indexY="+str(indexY)+" compare with indexX="+str(indexX)+" indexY="+str(indexY+1))
    # print("*********************")
    for indexY in range(regionStartY, regionEndY + 1):
        for indexX in range(regionStartX, regionEndX):
            value1 = float(inputImageGray[indexX, indexY])
            value2 = float(inputImageGray[indexX+1, indexY])
            temp = pow((value1 - value2), 2)
            edgeValues[index, 0] = temp
            index = index + 1
            # print("indexX="+str(indexX)+" indexY="+str(indexY)+" compare with indexX="+str(indexX+1)+" indexY="+str(indexY))
    return np.var(edgeValues)


def pointIndexToCoordinate(imageRow, imageCol, pointIndex):
    """
    功能：根据点位编号获取坐标
        输入：imageRow：原图的行数
              imageCol：原图的列数
        输出：返回该点在原图中的坐标值
    """
    rowIndex = int(pointIndex / imageCol)
    colIndex = pointIndex % imageCol
    return rowIndex, colIndex


def maxPixel(a, b):
    """
    功能：比两数大小
        输入：两个需要对比的数
        输出：两数中的最大值
    """
    return a if a >= b else b

# 进行gFunction运算
def gFunc(p, q, imageType="intensityImage", m = 10, variance = 1.0):
    # if variance == 0:   variance = 0.5
    # B = 1.0 / (2 * pow(variance, 2))
    B = 0.005
    value = 0.0
    if imageType == "intensityImage":           # 如果是 灰度 图像
        difference = float(p) - float(q)
        value = m * exp(-1 * B * pow(difference, 2))
    elif imageType == "edgeImage":              # 如果是 Edge 图像
        value = m * exp(-1 * B * pow(maxPixel(p, q), 2))
    # print("         B = "+str(B) + " p="+str(p)+" q=" + str(q) + " value=" + str(value))
    return value


def getBinaryCostByWaggoner(nextOriginalImage, lastHumanLabeledRGB, lastLabeled, type = "intensityImage", edgeOriginalImage = np.zeros((1,1)) ,neighborLength = 9, infiniteCost = 100):
    """
    功能：获得label的BinaryCost
        输入：nextOriginalImage：上一张原始图像
              lasthumanLabeledRGB：人为标注的RGB图像
              lastLabeled:上一张的标记结果
              type：处理图像类型（"intensityImage"或"edgeImage"）（可选，默认为"intensityImage"）
              edgeOriginalImage：边界图像，如果type == "edgeImage",此项才有用
              neighborLength: 邻域边界长度，（可选，必须为单数，默认为9）
        输出: binaryCost，m * 3(m 是边的数目，第一列边起点，第二列是边终点，第三列是边权值)
    """
    # 对图像进行转换，如果是彩色图像则转换为灰度图像
    inputImageGray = np.zeros(nextOriginalImage.shape)
    if nextOriginalImage.ndim == 3:
        inputImageGray = cv.cvtColor(nextOriginalImage, cv.COLOR_BGR2GRAY)
    elif nextOriginalImage.ndim == 2:
        inputImageGray = nextOriginalImage

    # 初始化边（m*3），每一行对应一条边，前两个值对于边的起点和终点，最后一个对应边的权值
    inds = np.arange(inputImageGray.size).reshape(inputImageGray.shape)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.zeros((horz.shape[0]+vert.shape[0], 3))
    edges[:, :-1] = np.vstack([horz, vert]).astype(np.int32)

    # 构建区域邻接图
    rag = graph.rag_mean_color(lastHumanLabeledRGB, lastLabeled)
    for index in range(0, edges.shape[0]):
        startPoint = edges[index][0].astype(np.int32)
        endPoint = edges[index][1].astype(np.int32)
        startRowIndex, startColIndex = pointIndexToCoordinate(inputImageGray.shape[0], inputImageGray.shape[1], startPoint)
        endRowIndex, endColIndex = pointIndexToCoordinate(inputImageGray.shape[0], inputImageGray.shape[1], endPoint)

        if type == "intensityImage":
            edges[index][2] = gFunc(inputImageGray[startRowIndex, startColIndex], inputImageGray[endRowIndex, endColIndex], imageType=type, m=1)
        elif type == "edgeImage":
            edges[index][2] = gFunc(edgeOriginalImage[startRowIndex, startColIndex], edgeOriginalImage[endRowIndex, endColIndex], imageType=type, m=1)

    return edges.astype(np.int32)

