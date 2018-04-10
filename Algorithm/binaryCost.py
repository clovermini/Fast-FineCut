# Function：calculate binaryCost
# Method：
#       1. Provide getBinaryCost functions
#       2. Provide gFunc functions
# Writer：Boyuan Ma，Time：2017-2-24
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
    Function：Obtain the Binary Cost of this layer to be segmented with the method proposed in fast-fine cut algorithm
        Input：nextOriginalImage: Last layer's original image
              lasthumanLabeledRGB：RGB image labeled by human of last layer's image
              lastLabeled: Labeled image of last layer's image
              lastNum：The label number of the result of the last layer
              unaryCostMatrix：Unary energy value matrix
              type：The type of image need to handled（"intensityImage" or "edgeImage"）（optional，default "intensityImage"）
              edgeOriginalImage：Border image, useful when type == "edgeImage"
              neighborLength: Neighbor boundary length, (optional, must be singular, default is 9)
              infiniteCost：Infinity setting, default 100
              KCost：K value of binary term, default 3
        Ouput: binaryCost，m * 3(m presents the number of edges, the first column is the beginning of edge, the second column is the end of the edge, the third column is the edge weight)
    """
    # Convert the image to a grayscale image if it is a color image
    inputImageGray = np.zeros(nextOriginalImage.shape)
    if nextOriginalImage.ndim == 3:
        inputImageGray = cv.cvtColor(nextOriginalImage, cv.COLOR_BGR2GRAY)
    elif nextOriginalImage.ndim == 2:
        inputImageGray = nextOriginalImage

    # Initialize edges (m*3), each row corresponds to one edge,
    # the first two values are for the start and end of the edge, and the last corresponding the weight of edge
    inds = np.arange(inputImageGray.size).reshape(inputImageGray.shape)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.zeros((horz.shape[0]+vert.shape[0], 3))
    edges[:, :-1] = np.vstack([horz, vert])    # np.vstack 矩阵的合并
    edges = edges.astype(np.int32)

    # Build area adjacency map
    rag = graph.rag_mean_color(lastOrigianlImage, lastLabeled)

    # Sets the weight of the area beyond boundingRegion to infinite
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

    # Set the weight of the area beside boundingRegion to a lower value combining the images of the next layer
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

    # Set the weight of the area beside boundingRegion to a lower value combining the images of the last layer
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

        if returnLabel[rowIndex, colIndex] > binaryCost[index , 2]:
            returnLabel[rowIndex, colIndex] = binaryCost[index , 2]

        rowIndex, colIndex = pointIndexToCoordinate(nextOriginalImage.shape[0], nextOriginalImage.shape[1], binaryCost[index, 1])

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
    index = np.int32(0)
    for indexX in range(regionStartX, regionEndX + 1):
        for indexY in range(regionStartY, regionEndY):
            value1 = float(inputImageGray[indexX, indexY])
            value2 = float(inputImageGray[indexX, indexY + 1])
            temp = float(pow((value1 - value2), 2))
            edgeValues[index, 0] = temp
            index = index + 1

    for indexY in range(regionStartY, regionEndY + 1):
        for indexX in range(regionStartX, regionEndX):
            value1 = float(inputImageGray[indexX, indexY])
            value2 = float(inputImageGray[indexX+1, indexY])
            temp = pow((value1 - value2), 2)
            edgeValues[index, 0] = temp
            index = index + 1

    return np.var(edgeValues)


def pointIndexToCoordinate(imageRow, imageCol, pointIndex):
    """
    Function：Obtain the coordinates based on point number
        Input：imageRow：row number of original image
              imageCol：column number of original image
        Output：Returns the coordinates of the point in the original image
    """
    rowIndex = int(pointIndex / imageCol)
    colIndex = pointIndex % imageCol
    return rowIndex, colIndex


def maxPixel(a, b):
    """
    Function：Compare the size of two numbers
        Input：number a and b
        Ouput：the max one of a and b
    """
    return a if a >= b else b

# Perform gFunction operation
def gFunc(p, q, imageType="intensityImage", m = 10, variance = 1.0):
    # if variance == 0:   variance = 0.5
    # B = 1.0 / (2 * pow(variance, 2))
    B = 0.005
    value = 0.0
    if imageType == "intensityImage":           # if the image type equals intensityImage
        difference = float(p) - float(q)
        value = m * exp(-1 * B * pow(difference, 2))
    elif imageType == "edgeImage":              # if the image type equals edgeImage
        value = m * exp(-1 * B * pow(maxPixel(p, q), 2))
    return value


def getBinaryCostByWaggoner(nextOriginalImage, lastHumanLabeledRGB, lastLabeled, type = "intensityImage", edgeOriginalImage = np.zeros((1,1)) ,neighborLength = 9, infiniteCost = 100):
    """
    Function：Obtain the Binary Cost of this layer to be segmented with waggoner's method
        Input：nextOriginalImage：Last layer's original image
              lasthumanLabeledRGB：RGB image labeled by human of last layer's image
              lastLabeled: Labeled image of last layer's image
              type：The type of image need to handled（"intensityImage" or "edgeImage"）（optional，default "intensityImage"）
              edgeOriginalImage：Border image, useful when type == "edgeImage"
              neighborLength: Neighbor boundary length, (optional, must be singular, default is 9)
        Ouput: binaryCost，m * 3(m presents the number of edges, the first column is the beginning of edge, the second column is the end of the edge, the third column is the edge weight)
    """

    # Convert the image to a grayscale image if it is a color image
    inputImageGray = np.zeros(nextOriginalImage.shape)
    if nextOriginalImage.ndim == 3:
        inputImageGray = cv.cvtColor(nextOriginalImage, cv.COLOR_BGR2GRAY)
    elif nextOriginalImage.ndim == 2:
        inputImageGray = nextOriginalImage

    # Initialize edges (m*3), each row corresponds to one edge,
    # the first two values are for the start and end of the edge, and the last corresponding the weight of edge
    inds = np.arange(inputImageGray.size).reshape(inputImageGray.shape)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.zeros((horz.shape[0]+vert.shape[0], 3))
    edges[:, :-1] = np.vstack([horz, vert]).astype(np.int32)

    # Build area adjacency map
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

