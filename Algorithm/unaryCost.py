# Function：calculate the unaryCost
# Method：
#       1. provide getUnaryCost function
#       2. provide getboundingRegion function
# Writer：Boyuan Ma，Time：2017-2-24
import numpy as np
from .propagationLabel import getNumberFromLabel, getEdgesFromLabel, getRGBLabel
import matplotlib.pyplot as plt
import cv2 as cv

def getUnaryCost(lastLabeled, lastSegment, return_boundingRegion = False, kernel=cv.MORPH_ELLIPSE, morphKernelNum = 10, infiniteCost=100):
    """
    Function：Obtain the unary cost of label
        Input：lastLabeled：label of last image
              lastSegment：segmentation result of last image(can be RGB image or grayscale image)
              return_boundingRegion: Whether to return boundingRegion, if yes, increase the boundary of each particle's boundingRegion in the original RGB image (optional, default is no)
              kernel: Expansion operation structure element shape (optional, defaults to cv.MORPH_ELLIPSE)
              morphKernelNum：Expansion operation structure element size (optional, defaults to 10)
              infiniteCost: infinite (optional, default is 100)
        Output: unaryCostMatrix：three-dimensional matrix of unary term (M * N * partilrNum) (M is the number of matrix rows, N is the number of matrix columns)
              bounndingRegion：the value of pixels belongs to the bounding region are 255, others are 0 (optional)
    """

    boundingRegion = np.zeros((lastLabeled.shape[0], lastLabeled.shape[1], 3)).astype(np.int32)

    if lastSegment.ndim == 3:
        boundingRegion = lastSegment
    elif lastSegment.ndim == 2:         # Change to RGB image if it is grayscale image
        boundingRegion = cv.cvtColor(lastSegment,cv.COLOR_GRAY2RGB)

    num = getNumberFromLabel(lastLabeled)               # Get the total number of labels
    unaryCostMatrix = np.ones((lastLabeled.shape[0], lastLabeled.shape[1], num), dtype=np.int32) * infiniteCost
    for index in range(1, num + 1):
        boundingRegionForIndex = getBoundingRegion(lastLabeled, index, kernel=kernel, morphKernelNum=morphKernelNum)
        if return_boundingRegion:
            edges = getEdgesFromLabel(boundingRegionForIndex, 8)
            boundingRegion[edges == 255] = [255, 255, 255]
        unaryCostMatrix[boundingRegionForIndex == 255, index-1] = 0

    if return_boundingRegion:
        return unaryCostMatrix, boundingRegion
    else:
        return unaryCostMatrix


def getBoundingRegion(labeled, indexNum, kernel=cv.MORPH_ELLIPSE, morphKernelNum=10):
    """
    Function：Get the boundingRegion of a label in the label image
        Input：labeled：Label matrix
              indexNum：label index
              kernel: Expansion operation structure element shape (optional, defaults to cv.MORPH_ELLIPSE)
              morphKernelNum：Expansion operation structure element size (optional, defaults to 10)
        Output: dilateI，the value of pixels belongs to the bounding region are 255, others are 0 (optional)
    """
    kernel = cv.getStructuringElement(kernel, (morphKernelNum, morphKernelNum))
    tempI = np.zeros(labeled.shape)
    tempI[labeled != indexNum] = 0
    tempI[labeled == indexNum] = 255
    dilateI = cv.dilate(tempI, kernel)
    return dilateI

