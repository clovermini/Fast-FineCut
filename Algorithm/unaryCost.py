# 功能：计算unaryCost
# 方法：
#       1. 提供getUnaryCost函数
#       2. 提供获取boundingRegion函数
#       2. 提供获取boundingRegion函数
# 编写人：马博渊，时间：2017-2-24
import numpy as np
from .propagationLabel import getNumberFromLabel, getEdgesFromLabel, getRGBLabel
import matplotlib.pyplot as plt
import cv2 as cv

def getUnaryCost(lastLabeled, lastSegment, return_boundingRegion = False, kernel=cv.MORPH_ELLIPSE, morphKernelNum = 10, infiniteCost=100):
    """
    功能：获得label的unaryCost
        输入：lastLabeled：上一层图像的标记
              lastSegment：上一层分割后的图像(可为彩色，也可为灰度图像)
              return_boundingRegion: 是否返回boundingRegion，是的话在原RGB图中增加每个粒子的boundingRegion的边界（可选，默认为否）
              kernel: 膨胀操作结构元素形状（可选，默认为cv.MORPH_ELLIPSE）
              morphKernelNum：膨胀操作结构元素大小（可选，默认为10）
              infiniteCost: 代价为无穷的大小，（可选，默认为1000000）
        输出: unaryCostMatrix：一元项三维矩阵（M * N * partilrNum）(M为矩阵行数，N为矩阵列数)
              bounndingRegion：属于该region像素值为255，不属于为0（可选）
    """

    boundingRegion = np.zeros((lastLabeled.shape[0], lastLabeled.shape[1], 3)).astype(np.int32)

    if lastSegment.ndim == 3:           # 若为彩色图像，则直接赋值
        boundingRegion = lastSegment
    elif lastSegment.ndim == 2:         # 若为灰度图像，则转换为彩色图像
        boundingRegion = cv.cvtColor(lastSegment,cv.COLOR_GRAY2RGB)

    num = getNumberFromLabel(lastLabeled)               # 获取总编号数目
    # print("label.shape="+str(lastLabeled.shape)+ ", num="+str(num))
    unaryCostMatrix = np.ones((lastLabeled.shape[0], lastLabeled.shape[1], num), dtype=np.int32) * infiniteCost
    # unaryCostMatrix = np.zeros((lastLabeled.shape[0], lastLabeled.shape[1], num), dtype=np.int32)
    for index in range(1, num + 1):
        boundingRegionForIndex = getBoundingRegion(lastLabeled, index, kernel=kernel, morphKernelNum=morphKernelNum)
        if return_boundingRegion:
            edges = getEdgesFromLabel(boundingRegionForIndex, 8)
            boundingRegion[edges == 255] = [255, 255, 255]
        # unaryCostMatrix[boundingRegion == 255, index-1] = infiniteCost
        unaryCostMatrix[boundingRegionForIndex == 255, index-1] = 0

    if return_boundingRegion:
        return unaryCostMatrix, boundingRegion
    else:
        return unaryCostMatrix


def getBoundingRegion(labeled, indexNum, kernel=cv.MORPH_ELLIPSE, morphKernelNum=10):
    """
    功能：获得label中某个标签的boundingRegion
        输入：labeled：标签矩阵
              indexNum：需要求boundingRegion的某个标签
              kernel: 膨胀操作结构元素形状（可选，默认为cv.MORPH_ELLIPSE）
              morphKernelNum：膨胀操作结构元素大小（可选，默认为10）
        输出: dilateI，属于该boundingRegion像素值为255，不属于为0
    """
    kernel = cv.getStructuringElement(kernel, (morphKernelNum, morphKernelNum))
    tempI = np.zeros(labeled.shape)
    tempI[labeled != indexNum] = 0
    tempI[labeled == indexNum] = 255
    dilateI = cv.dilate(tempI, kernel)
    return dilateI

