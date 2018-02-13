#  -*- coding:utf-8 -*-  
# 功能：本工程的主函数
import numpy as np
import cv2 as cv
from .unaryCost import getUnaryCost
from .binaryCost import getBinaryCostFromUnary,getBinaryCostByWaggoner
from .binaryCost import addComparation
from .pairWise import getPairWiseMatrix
from .propagationLabel import reviseLabel,getEdgesFromLabel,denoiseByArea

from skimage import morphology
from skimage.measure import label
from pygco import cut_from_graph

def propagationSegment(lastSegmentImageAddress, nextOriginalImageAddress, nextSegmentAddress, boundingLength = 18, infiniteCost = 100, KCost = 3):
  """
  传播分割主函数
  :param lastSegmentImageAddress: 上一层图片的分割结果存放地址
  :param nextOriginalImageAddress: 本层待分割图片的原图地址
  :param nextSegmentAddress: 本层图片分割结果存放地址
  :param boundingLength: 边界区域长度
  :param infiniteCost: 无穷大权值的设定值，默认为100
  :param KCost: 二元项中K值设定
  :return: 返回分割结果
  """

  # 读取图像
  tempLastSegment = cv.imread(lastSegmentImageAddress)
  if tempLastSegment is None:
      return "上一层图像路径错误"

  tempNextOriginal = cv.imread(nextOriginalImageAddress)
  if tempNextOriginal is None:
      return "本层图像路径错误"

  # 获取图像横纵坐标的数目，获取二维矩阵的第一维和第二为的长度
  rowNumber = tempLastSegment.shape[0]
  colNumber = tempLastSegment.shape[1]

  # 对上一层已分割图像判断，若为彩色转成灰度图像，若为灰度则不变
  lastSegment = np.zeros((rowNumber, colNumber))
  if tempLastSegment.ndim == 3:
      lastSegment = cv.cvtColor(tempLastSegment, cv.COLOR_BGR2GRAY)
  elif tempLastSegment.ndim == 2:
      lastSegment = tempLastSegment

  # 对本层待分割图像判断，若为彩色转成灰度图像，若为灰度则不变
  nextOriginal = np.zeros((rowNumber, colNumber))
  if tempNextOriginal.ndim == 3:
      nextOriginal = cv.cvtColor(tempNextOriginal, cv.COLOR_BGR2GRAY)
  elif tempNextOriginal.ndim == 2:
      nextOriginal = tempNextOriginal

  if nextOriginal.shape != lastSegment.shape:
      return "上一层图像和本层图像维度不一样，错误"


  # 对上一层图像进行标记
  uniqueNum = np.unique(lastSegment)
  lastNumber = 0
  lastLabeled = np.zeros((rowNumber, colNumber))
  if len(uniqueNum) == 2:    # 二值图像
      (lastLabeled, lastNumber) = label(lastSegment, background=255, neighbors=4, return_num=True)
      lastLabeled = reviseLabel(lastLabeled)
  elif len(uniqueNum) > 2:
      (lastLabeled, lastNumber) = label(lastSegment, neighbors=4, return_num=True)
  lastLabeled = lastLabeled.astype(np.int32)

  # 计算一元项 unaryCostMatrix 区域项
  unaryCostMatrix, boundingRegion = getUnaryCost(lastLabeled, lastSegment, return_boundingRegion=True, morphKernelNum=boundingLength, infiniteCost=infiniteCost)

  # 计算二元项所需的 edgeImage
  blurredGaussian = cv.GaussianBlur(nextOriginal, (3, 3), 0)         # 高斯滤波
  imgThreshMean = cv.adaptiveThreshold(blurredGaussian, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 4)
  imgThreshMean = denoiseByArea(imgThreshMean, 300, neighbors=8)
  tempNextEdge = morphology.skeletonize(imgThreshMean / 255) * 255

  # Fast-FineCut二元项，使用 edgeImage
  binaryCostMatrix = getBinaryCostFromUnary(nextOriginal, lastSegment, lastLabeled, lastNumber, unaryCostMatrix, type="edgeImage", edgeOriginalImage=tempNextEdge, infiniteCost=infiniteCost, KCost=KCost)
  # waggoner二元项
  # binaryCostMatrix = getBinaryCostByWaggoner(nextOriginal, lastSegment, lastLabeled, type="edgeImage", edgeOriginalImage=tempNextEdge, infiniteCost=infiniteCost)

  # 计算 pairWiseMatrix
  pairWiseMatrix = getPairWiseMatrix(lastSegment, lastLabeled)

  # 图割
  result_graph = cut_from_graph(binaryCostMatrix, unaryCostMatrix.reshape(-1, lastNumber), pairWiseMatrix, algorithm="swap")
  nextLabeled = result_graph.reshape(nextOriginal.shape)
  nextSegment = getEdgesFromLabel(nextLabeled)

  # 查看中间结果
  # plt.subplot(231),plt.imshow(tempNextEdge,cmap="gray"),plt.title("nextEdges")
  # plt.subplot(232),plt.imshow(np.uint8(boundingRegion)),plt.title("boundingRegion")
  # #plt.subplot(232),plt.imshow(tempNextEdge,cmap="gray"),plt.title("nextEdges")
  # # plt.subplot(333),plt.imshow(unaryCost),plt.title("unaryCost")
  # plt.subplot(233),plt.imshow(addComparation(nextOriginal, binaryCostMatrix)),plt.title("binaryCostMatrix")
  # #plt.subplot(234),plt.imshow(ragShow),plt.title("rag")
  # plt.subplot(234),plt.imshow(nextOriginal,cmap="gray"),plt.title("Original")
  # plt.subplot(235),plt.imshow(nextLabeled),plt.title("nextLabeled")
  # plt.subplot(236),plt.imshow(nextSegment,cmap="gray"),plt.title("result")
  # plt.show()

  # 导出图像
  try:
      cv.imwrite(nextSegmentAddress, nextSegment)
  except (cv.error, Exception):
      return "图片保存错误"
  return nextSegment






