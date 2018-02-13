# 功能：计算pairWise矩阵，
# 原理：上一层不相邻的两个编号在下一层的相邻概率小
# 方法：
#       1. 提供getPairWise函数
#       2. 提供获取boundingRegion函数
#       2. 提供获取boundingRegion函数
# 编写人：马博渊，时间：2017-2-24
import numpy as np
import cv2 as cv
from .propagationLabel import getRGBLabel, getNumberFromLabel
from skimage.future import graph


def getPairWiseMatrix(lastOriginalImage, lastLabeled, diagonalNum = -15, adjacentNum = -8):
    """
    功能：获得label中某个的PairWiseMatrix
        输入：lastOriginalImage：上一张原始图像
              diagonalNum：对角线元素大小（可选，默认为-15）
              adjacentNum：相邻粒子的权重（可选，默认为-8）
        输出: pairWiseMatrix，返回该矩阵，对角线为diagonalNum，相邻的设置为adjacentNum，不相邻设置为0
    """
    labelNum = getNumberFromLabel(lastLabeled)         # 编号问题（取最大数可能有问题）
    pairWiseMatrix = diagonalNum * np.eye(labelNum, dtype=np.int32)
    # print(pairWiseMatrix.shape)
    rag = graph.rag_mean_color(lastOriginalImage, lastLabeled)
    for index in range(0, labelNum):
        # print(str(index)+" 节点为"+str(index+1))
        if index+1 in lastLabeled:
            neighbors = rag.neighbors(index+1)
            # print(neighbors)
            for label in neighbors:
                pairWiseMatrix[index, label-1] = adjacentNum
                pairWiseMatrix[label-1, index] = adjacentNum
    # print(pairWiseMatrix)
    return pairWiseMatrix

