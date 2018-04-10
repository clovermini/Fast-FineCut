# Function：All functions related to label
# Writer：Boyuan Ma，Time：2017-2-24

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
    Function：Normalize Label
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
    Function： Remove noises with area less than a certain threshold
        Input： grayImage: grayscale image
              areaThresh:  threshold of area
              neighbors: 4 Neighbors or 8 Neighbors (optional, default 4 Neighbors)
        Output: grayImage without noises
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
    Function：Output a full-contrast image with the edges of the image green, the GroundTruth red, and the boundingRegion purple
        Output：RGB results
        :param segLabeled:
    """
    returnImage = np.zeros((segLabeled.shape[0], segLabeled.shape[1], 3))
    segEdge = getEdgesFromLabel(segLabeled, neighbors=4)
    truEdge = getEdgesFromLabel(truLabeled, neighbors=4)

    for indexX in range(0, segLabeled.shape[0]):
        for indexY in range(0, segLabeled.shape[1]):
            if segEdge[indexX, indexY] == 255:
                returnImage[indexX, indexY] = [0, 255, 0]
            elif truEdge[indexX, indexY] == 255 and segEdge[indexX, indexY] != 255:
                returnImage[indexX, indexY] = [0, 0, 255]
            elif (np.array(boundingImage[indexX, indexY]) == [255, 255, 255]).all() and segEdge[indexX, indexY] != 255:
                returnImage[indexX, indexY] = [238, 122, 233]

    return returnImage

def addComparation(originalImage, comparation, value):
    """
    Function：Add the result of the comparison to original image
        Input： originalImage: original image
               comparation: Comparison image (the same size as original image)
               value: the pixel value after increasing the comparison image
        Output：Comparison image (RGB)
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

def getDistanceFromGroundTruthPoint(groundTruthEdge, indexX, indexY, neighborLength = 60):
    """
    Function：Calculate the distance between the detected edge point and its nearest edge point
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

    indexXX = 0
    indexYY=0
    for indexGroundX in range(regionStartRow, regionEndRow):
        for indexGroundY in range(regionStartCol, regionEndCol):
            if groundTruthEdge[indexGroundX, indexGroundY] == 255:
                tempDisance = np.sqrt(pow(indexGroundX - indexX,2)+pow(indexGroundY - indexY,2))
                if tempDisance < distance:
                    distance = tempDisance
                    indexXX = indexGroundX
                    indexYY = indexGroundY
    return distance

def evaluateMeritForEdge(segmentEdge, groundTruthEdge, constIndex = 0.1):
    """
    Function：Use figure of merit to evaluate the edge detection quality of proposed method

    """
    numSeg = np.sum(segmentEdge[segmentEdge == 255])/255
    numGround = np.sum(groundTruthEdge[groundTruthEdge == 255])/255
    maxNum = numSeg

    if numSeg < numGround:  maxNum = numGround
    temp = 0.0
    for indexX in range(0, segmentEdge.shape[0]):
        for indexY in range(0, segmentEdge.shape[1]):
             if segmentEdge[indexX, indexY] == 255:
                distance = getDistanceFromGroundTruthPoint(groundTruthEdge, indexX, indexY)
                temp = temp + 1 / (1 + constIndex * pow(distance,2))
    merit = (1.0 / maxNum) * temp
    return merit

def getEdgesFromLabel(labeled, neighbors=4):
    """
    Function：Obtain the border image based on label
        Input： labeled: label image
               neighbor: 4 Neighbors or 8 Neighbors, (optional, default 4 Neighbors)
        Algorithm：If there is a pixel in the neighborhood of a certain pixel that is different from its own label, the pixel is a boundary pixel, set to 255, otherwise it is 0.
        Output： edgeImage (edgeImage == 0 or edgeImage == 255)
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

def getNumberFromLabel(labeled):
    """
    Function：Get the number of particles from the label
        Input： labeled: label image
        Algorithm： Taking into account the possible discontinuity of the label, first unique then seek length
        Output： Number of particles in the label map
    """
    num = len(np.unique(labeled))
    if 0 in labeled:
        num = num - 1
    return num

