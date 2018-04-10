# Function：calculate pairWise matrix
# Principle：The two Non-adjacent numbers on the last layer are less likely to be adjacent in the next layer.
# Method：provide getPairWise function
# Writer：Boyuan ma. Time：2017-2-24
import numpy as np
import cv2 as cv
from .propagationLabel import getRGBLabel, getNumberFromLabel
from skimage.future import graph


def getPairWiseMatrix(lastOriginalImage, lastLabeled, diagonalNum = -15, adjacentNum = -8):
    """
    Function：Obtaion one PairWiseMatrix in a label image
        Input：lastOriginalImage：last layer's original image
              diagonalNum：Diagonal element size (optional, default -15)
              adjacentNum：Weights of adjacent particles (optional, default -8)
        Output: pairWiseMatrix，Returns this matrix with a diagonalNum, the adjacent setting is adjacentNum, and the nonadjacent setting is 0.
    """
    labelNum = getNumberFromLabel(lastLabeled)         # Numbering problem (the maximum number may have a problem)
    pairWiseMatrix = diagonalNum * np.eye(labelNum, dtype=np.int32)
    rag = graph.rag_mean_color(lastOriginalImage, lastLabeled)
    for index in range(0, labelNum):
        if index+1 in lastLabeled:
            neighbors = rag.neighbors(index+1)
            for label in neighbors:
                pairWiseMatrix[index, label-1] = adjacentNum
                pairWiseMatrix[label-1, index] = adjacentNum
    return pairWiseMatrix

