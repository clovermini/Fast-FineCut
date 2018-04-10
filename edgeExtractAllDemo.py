import cv2 as cv
import numpy as np
from Algorithm.propagationLabel import getEdgesFromLabel,evaluateMeritForEdge
from skimage.measure import label
import matplotlib.pyplot as plt
from skimage import feature,morphology
import time
import os

def meanGray(gray, threshold):
    lowSum = 0.0
    lowNum = 0
    highNum = 0
    highSum = 0.0
    for i in range(0, gray.shape[0]):
        for j in range(0, gray.shape[1]):
            if (gray[i, j] > threshold):
                highSum += gray[i, j]
                highNum = highNum + 1
            elif (gray[i, j] <= threshold):
                lowSum += gray[i, j]
                lowNum = lowNum + 1
    meanhigh = highSum*1.0 / highNum
    meanlow = lowSum*1.0 / lowNum
    L1 = (meanhigh + meanlow) / 2
    return (meanhigh, meanlow, L1)

def iteractiveMethod(gray):
    T_max = int(gray.max())
    T_min = int(gray.min())
    L0 = (T_max + T_min)*1.0 / 2

    threshold = 0
    while(1):
        (meanhigh, meanlow, L1) = meanGray(gray, L0)
        if(L1 == L0):
            threshold = L0
            break
        L0 = L1
    return threshold

def showHist(grayImage, figureName):
    plt.figure(figureName)
    hist = cv.calcHist([grayImage], [0], None, [256], [0, 256])
    plt.subplot(121), plt.title(u"Original"), plt.imshow(grayImage, cmap="gray")
    plt.subplot(122), plt.title(u"Grayscale histogram"), plt.plot(hist)
    plt.show()

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



if __name__ == "__main__":

    # *************************************** Init ******************************************************
    slices = ['001', '002', '003', '004']
    numSlices = len(slices)
    otsuF = np.zeros(numSlices)
    iteractiveF = np.zeros(numSlices)
    cannyF = np.zeros(numSlices)
    meanF = np.zeros(numSlices)
    gaussianF = np.zeros(numSlices)
    F_average = np.zeros(5)

    addressProject = os.path.join(os.getcwd(), "images")    # Image directory
    nextOriginalImageAddress = addressProject + "\\Original\\"  # Original images directory
    nextHumanLabeledRGBAddress = addressProject + "\\GroundTruth\\"  # Ground truth images directory
    resultsAddress = addressProject + "\\Results\\Others\\"  # Storage directory of segmentation results

    for index in range(0, numSlices):     # A total of 5 series of images
        nextOriginalImage = cv.cvtColor(cv.imread(nextOriginalImageAddress+slices[index]+".tif"), cv.COLOR_BGR2GRAY)
        nextHumanLabeledRGB = cv.imread(nextHumanLabeledRGBAddress+slices[index]+".tif")
        rowNumber = nextHumanLabeledRGB.shape[0]
        colNumber = nextHumanLabeledRGB.shape[1]
        nextHumanLabeledGray = np.zeros((rowNumber, colNumber))
        nextHumanLabeledGray = cv.cvtColor(nextHumanLabeledRGB, cv.COLOR_BGR2GRAY)
        (nextHumanLabeled, nNumber) = label(nextHumanLabeledGray, neighbors=4, return_num=True)
        nextHumanLabeled = nextHumanLabeled.astype(np.int32)
        nextGroundTruthEdge = np.zeros((rowNumber, colNumber))
        nextGroundTruthEdge = getEdgesFromLabel(nextHumanLabeled)

        nextGroundTruthEdge_fc = morphology.skeletonize(nextGroundTruthEdge / 255) * 255
        cv.imwrite(resultsAddress+slices[index]+"-gt.tif", nextGroundTruthEdge_fc)
        print("segment image ："+slices[index])

        start_time = time.time()

        # ********************************* Otsu method *****************************************
        threshold, imgOtsu = cv.threshold(nextOriginalImage, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # ********************************* Iterative threshold method ****************************************
        threshold_iM = iteractiveMethod(nextOriginalImage)
        T_two, imgIteractiveSeg = cv.threshold(nextOriginalImage, threshold_iM, 255, cv.THRESH_BINARY_INV)  # 阈值化处理，阈值为：155

        # ********************************* Sobel method ****************************************
        # x = cv.Sobel(nextOriginalImage, cv.CV_16S, 1, 0)
        # y = cv.Sobel(nextOriginalImage, cv.CV_16S, 0, 1)
        # absX = cv.convertScaleAbs(x)   # turn to uint8
        # absY = cv.convertScaleAbs(y)
        # imgSobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        #
        # ******************************* Adaptive threshold method **************************************

        blurredGaussian = cv.GaussianBlur(nextOriginalImage, (3, 3), 0) # Gaussian filter
        imgThreshMean = cv.adaptiveThreshold(blurredGaussian, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 4)

        # imgThreshMean = denoiseByArea(imgThreshMean, 300, neighbors=4)
        # imgThreshMean = morphology.skeletonize(imgThreshMean / 255) * 255
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C：Calculate the neighborhood weighted average as a threshold

        imgThreshGaussian = cv.adaptiveThreshold(blurredGaussian, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 4)
        # imgThreshGaussian = denoiseByArea(imgThreshGaussian, 300, neighbors=4)
        # imgThreshGaussian = morphology.skeletonize(imgThreshGaussian / 255) * 255

        # ********************************** Canny method *****************************************
        imgCanny = cv.Canny(blurredGaussian, 150, 200)

        # ********************************** Dilation and erosion *****************************************
        # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
        # imgEroded = cv.erode(nextOriginalImage, kernel)             # erosion
        # imgDilated = cv.dilate(nextOriginalImage, kernel)           # Dilation
        # imgClosed = cv.morphologyEx(nextOriginalImage, cv.MORPH_CLOSE, kernel)  # close operation
        # imgOpened = cv.morphologyEx(nextOriginalImage, cv.MORPH_OPEN, kernel)   # open operation
        #
        # edgeSegI1 = cv.subtract(imgDilated, nextOriginalImage)
        # edgeSegI2 = cv.subtract(nextOriginalImage, imgEroded)
        # edgeSegI3 = cv.subtract(imgDilated, imgEroded)
        # edgeSegI4 = cv.subtract(nextOriginalImage, imgOpened)
        # edgeSegI5 = cv.subtract(imgClosed, nextOriginalImage)
        # threshold, edgeSegI5 = cv.threshold(edgeSegI5, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # edgeSegI6 = cv.subtract(imgClosed, imgOpened)
        # # showHist(edgeSegI6, "edgeSegI6")
        # #threshold, edgeSegI6 = cv.threshold(edgeSegI6, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        #
        # edgeSegI7 = cv.subtract(imgDilated, imgClosed)
        # edgeSegI8 = cv.subtract(imgOpened, imgEroded)
        # edgeSegI9 = edgeSegI1.copy()
        # for i in range(edgeSegI8.shape[0]):
        #     for j in range(edgeSegI8.shape[1]):
        #         if edgeSegI2[i, j] < edgeSegI1[i, j]:
        #             edgeSegI9[i, j] = edgeSegI2[i, j]

        end_time = time.time()
        print("it cost ", str(end_time-start_time), "sec")

        # # # ********************************* evaluate ****************************************

        print(u' figure of merit：')
        # *************************Otsu*******************************
        imgOtsu_fc = morphology.skeletonize(imgOtsu / 255) * 255
        otsuF[index] = evaluateMeritForEdge(imgOtsu_fc, nextGroundTruthEdge_fc)
        cv.imwrite(resultsAddress+slices[index]+"-Otsu.tif", imgOtsu_fc)
        print(' Otsu F='+str(otsuF[index]))
        F_average[0] += otsuF[index]
        #
        # ************************Iteractive****************************
        imgIteractiveSeg_fc = morphology.skeletonize(imgIteractiveSeg / 255) * 255
        iteractiveF[index] = evaluateMeritForEdge(imgIteractiveSeg_fc, nextGroundTruthEdge_fc)
        cv.imwrite(resultsAddress+slices[index]+"-Iteractive.tif", imgIteractiveSeg_fc)
        print(' Iteractive F='+str(iteractiveF[index]))
        F_average[1] += iteractiveF[index]
        #
        # ***********************Canny*****************************************
        imgCanny_fc = morphology.skeletonize(imgCanny / 255) * 255
        cannyF[index] = evaluateMeritForEdge(imgCanny_fc, nextGroundTruthEdge_fc)
        cv.imwrite(resultsAddress+slices[index]+"-Canny.tif", imgCanny_fc)
        print(' Canny F='+str(cannyF[index]))
        F_average[2] += cannyF[index]
        #
        #  *************************ThreshMean*********************************
        imgThreshMean_fc = morphology.skeletonize(imgThreshMean / 255) * 255
        meanF[index] = evaluateMeritForEdge(imgThreshMean_fc, nextGroundTruthEdge_fc)
        cv.imwrite(resultsAddress+slices[index]+"-adpativeMean.tif", imgThreshMean_fc)
        print(' adpativeMean F='+str(meanF[index]))
        F_average[3] += meanF[index]
        #
        # ************************** ThreshGaussian ********************************
        imgThreshGaussian_fc = morphology.skeletonize(imgThreshGaussian / 255) * 255
        gaussianF[index] = evaluateMeritForEdge(imgThreshGaussian_fc, nextGroundTruthEdge_fc)
        cv.imwrite(resultsAddress+slices[index]+"-gaussian.tif", imgThreshGaussian_fc)
        print(' gaussian F='+str(gaussianF[index]))
        F_average[4] += gaussianF[index]

    for av in range(0, 5):
        F_average[av] = F_average[av]/len(slices)
    print(' Otsu avF=', str(F_average[0]), ' Iteractive avF=', str(F_average[1]), ' Canny avF=', str(F_average[2]), ' adpativeMean avF=', str(F_average[3]), ' gaussian avF=', str(F_average[4]))

print("over")



