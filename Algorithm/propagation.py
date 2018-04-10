#  -*- coding:utf-8 -*-  
import numpy as np
import cv2 as cv
from .unaryCost import getUnaryCost
from .binaryCost import getBinaryCostFromUnary,getBinaryCostByWaggoner
from .binaryCost import addComparation
from .pairWise import getPairWiseMatrix
from .propagationLabel import reviseLabel, getEdgesFromLabel, denoiseByArea

from skimage import morphology
from skimage.measure import label
from pygco import cut_from_graph

def propagationSegment(lastSegmentImageAddress, nextOriginalImageAddress, nextSegmentAddress, algorithm = "ffc", boundingLength = 18, infiniteCost = 100, KCost = 3):
  """
  the main function of propagation segmentation
  :param lastSegmentImageAddress: The address of the segment result of last image
  :param nextOriginalImageAddress: The address of the original image to be segmented in this layer
  :param nextSegmentAddress: The storage address of segmentation results
  :param algorithm: Algorithm category Fast-FineCut -- "ffc"  Waggoner -- "wag"
  :param boundingLength: the length of bounding region, default 18
  :param infiniteCost:  Default 100, refers to infinity
  :param KCost: K value of binary term, default 3
  :return: Return the segmentation result
  """

  # read the image
  tempLastSegment = cv.imread(lastSegmentImageAddress)
  if tempLastSegment is None:
      return "The last image's path is wrong"

  tempNextOriginal = cv.imread(nextOriginalImageAddress)
  if tempNextOriginal is None:
      return "The path of this layer's image is wrong"

  # Gets the number of horizontal and vertical coordinates of the image
  rowNumber = tempLastSegment.shape[0]
  colNumber = tempLastSegment.shape[1]

  # Convert the last image to a grayscale image if it is a color image
  lastSegment = np.zeros((rowNumber, colNumber))
  if tempLastSegment.ndim == 3:
      lastSegment = cv.cvtColor(tempLastSegment, cv.COLOR_BGR2GRAY)
  elif tempLastSegment.ndim == 2:
      lastSegment = tempLastSegment

  # Convert this layered image to a grayscale image if it is a color image
  nextOriginal = np.zeros((rowNumber, colNumber))
  if tempNextOriginal.ndim == 3:
      nextOriginal = cv.cvtColor(tempNextOriginal, cv.COLOR_BGR2GRAY)
  elif tempNextOriginal.ndim == 2:
      nextOriginal = tempNextOriginal

  if nextOriginal.shape != lastSegment.shape:
      return "Error! The image of the previous layer and the image of the layer is different in dimension."


  # Label the last image
  uniqueNum = np.unique(lastSegment)
  lastNumber = 0
  lastLabeled = np.zeros((rowNumber, colNumber))
  if len(uniqueNum) == 2:    # Binary image
      (lastLabeled, lastNumber) = label(lastSegment, background=255, neighbors=4, return_num=True)
      lastLabeled = reviseLabel(lastLabeled)
  elif len(uniqueNum) > 2:
      (lastLabeled, lastNumber) = label(lastSegment, neighbors=4, return_num=True)
  lastLabeled = lastLabeled.astype(np.int32)

  # Calculate the unary item unaryCostMatrix and bounding region
  unaryCostMatrix, boundingRegion = getUnaryCost(lastLabeled, lastSegment, return_boundingRegion=True, morphKernelNum=boundingLength, infiniteCost=infiniteCost)

  # Calculate the edgeImage required for the binary item
  blurredGaussian = cv.GaussianBlur(nextOriginal, (3, 3), 0)
  imgThreshMean = cv.adaptiveThreshold(blurredGaussian, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 4)
  imgThreshMean = denoiseByArea(imgThreshMean, 300, neighbors=8)
  tempNextEdge = morphology.skeletonize(imgThreshMean / 255) * 255

  if algorithm == "ffc":
    # Fast-FineCut's binary term, use edgeImage
    binaryCostMatrix = getBinaryCostFromUnary(nextOriginal, lastSegment, lastLabeled, lastNumber, unaryCostMatrix, type="edgeImage", edgeOriginalImage=tempNextEdge, infiniteCost=infiniteCost, KCost=KCost)
  elif algorithm == "wag":
    # waggoner's binary term
    binaryCostMatrix = getBinaryCostByWaggoner(nextOriginal, lastSegment, lastLabeled, type="edgeImage", edgeOriginalImage=tempNextEdge, infiniteCost=infiniteCost)

  # calculate pairWiseMatrix
  pairWiseMatrix = getPairWiseMatrix(lastSegment, lastLabeled)

  # Graph-cut
  result_graph = cut_from_graph(binaryCostMatrix, unaryCostMatrix.reshape(-1, lastNumber), pairWiseMatrix, algorithm="swap")
  nextLabeled = result_graph.reshape(nextOriginal.shape)
  nextSegment = getEdgesFromLabel(nextLabeled)

  # Show the intermediate results
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

  # Export image
  try:
      cv.imwrite(nextSegmentAddress, nextSegment)
  except (cv.error, Exception):
      return "Picture is saved incorrectly"
  return nextSegment






