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
    plt.subplot(121), plt.title(u"原图"), plt.imshow(grayImage, cmap="gray")
    plt.subplot(122), plt.title(u"灰度直方图"), plt.plot(hist)
    plt.show()

def denoiseByArea(grayImage, areaThresh, neighbors = 8):
    """
    功能： 去掉面积小于一定阈值的噪声点
        输入： grayImage:灰度图像
              areaThresh: 面积阈值
              neighbors:4邻域或8邻域（可选，默认4邻域）
        输出: 去掉噪声的图
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

    # *************************************** 初始化 ******************************************************
    slices = ['001', '002', '003', '004']
    numSlices = len(slices)
    otsuF = np.zeros(numSlices)
    iteractiveF = np.zeros(numSlices)
    cannyF = np.zeros(numSlices)
    meanF = np.zeros(numSlices)
    gaussianF = np.zeros(numSlices)
    F_average = np.zeros(5)

    addressProject = os.path.join(os.getcwd(), "images")    # 分析图像所在目录
    nextOriginalImageAddress = addressProject + "\\Original\\"  # 原图所在地址
    nextHumanLabeledRGBAddress = addressProject + "\\GroundTruth\\"  # 真值图所在地址
    resultsAddress = addressProject + "\\Results\\Others\\"  # 分割结果保存地址

    for index in range(0, numSlices):     # 共11张系列图像
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
        print("编号："+slices[index]+"， 真值图像标记完成")

        start_time = time.time()

        # ********************************* Otsu 方法 *****************************************
        threshold, imgOtsu = cv.threshold(nextOriginalImage, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # ********************************* 迭代阈值法 ****************************************
        threshold_iM = iteractiveMethod(nextOriginalImage)
        T_two, imgIteractiveSeg = cv.threshold(nextOriginalImage, threshold_iM, 255, cv.THRESH_BINARY_INV)  # 阈值化处理，阈值为：155

        # ********************************* Sobel 方法 ****************************************
        # x = cv.Sobel(nextOriginalImage, cv.CV_16S, 1, 0)
        # y = cv.Sobel(nextOriginalImage, cv.CV_16S, 0, 1)
        # absX = cv.convertScaleAbs(x)   # 转回 uint8
        # absY = cv.convertScaleAbs(y)
        # imgSobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        #
        # ******************************* 自适应阈值化处理 **************************************
        # cv2.ADAPTIVE_THRESH_MEAN_C：计算邻域均值作为阈值
        blurredGaussian = cv.GaussianBlur(nextOriginalImage, (3, 3), 0) # 高斯滤波
        imgThreshMean = cv.adaptiveThreshold(blurredGaussian, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 4)

        # imgThreshMean = denoiseByArea(imgThreshMean, 300, neighbors=4)
        # imgThreshMean = morphology.skeletonize(imgThreshMean / 255) * 255
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C：计算邻域加权平均作为阈值

        imgThreshGaussian = cv.adaptiveThreshold(blurredGaussian, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 4)
        # imgThreshGaussian = denoiseByArea(imgThreshGaussian, 300, neighbors=4)
        # imgThreshGaussian = morphology.skeletonize(imgThreshGaussian / 255) * 255

        # ********************************** Canny 方法 *****************************************
        imgCanny = cv.Canny(blurredGaussian, 150, 200)

        # ********************************** 膨胀腐蚀 *****************************************
        # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
        # imgEroded = cv.erode(nextOriginalImage, kernel)             # 腐蚀
        # imgDilated = cv.dilate(nextOriginalImage, kernel)           # 膨胀
        # imgClosed = cv.morphologyEx(nextOriginalImage, cv.MORPH_CLOSE, kernel)  # 闭运算
        # imgOpened = cv.morphologyEx(nextOriginalImage, cv.MORPH_OPEN, kernel)   # 开运算
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
        # # # ********************************* plt 展示 ****************************************
        # plt.figure(1)
        # plt.subplot(231, title="OriginalImage")
        # plt.imshow(nextOriginalImage, cmap="gray")
        # plt.subplot(232, title="nextGroundTruthEdge")
        # plt.imshow(nextGroundTruthEdge, cmap="gray")
        # plt.subplot(233, title="iteractiveSeg")
        # plt.imshow(imgIteractiveSeg, cmap="gray")
        # plt.subplot(234, title="Otsu")
        # plt.imshow(imgOtsu, cmap="gray")
        # plt.subplot(235, title="ADAPTIVE_THRESH_GAUSSIAN")
        # plt.imshow(imgThreshGaussian, cmap="gray")
        # plt.subplot(236, title="Canny")
        # plt.imshow(imgCanny, cmap="gray")
        #
        # plt.figure(2)
        # plt.subplot(521), plt.imshow(nextOriginalImage, 'gray'), plt.title(u'原图', fontproperties='SimHei')
        # plt.subplot(522), plt.imshow(edgeSegI1, 'gray'), plt.title(u'膨胀型 I1=(f⊕b)-f', fontproperties='SimHei')
        # plt.subplot(523), plt.imshow(edgeSegI2, 'gray'), plt.title(u'腐蚀型 I2=f-(f⊝b)', fontproperties='SimHei')
        # plt.subplot(524), plt.imshow(edgeSegI3, 'gray'), plt.title(u'膨胀腐蚀型 I3=(f⊕b)-(f⊝b)', fontproperties='SimHei')
        # plt.subplot(525), plt.imshow(edgeSegI4, 'gray'), plt.title(u'开启型 I4=f-(fOb)', fontproperties='SimHei')
        # plt.subplot(526), plt.imshow(edgeSegI5, 'gray'), plt.title(u'闭合型 I5=(f.b)-f', fontproperties='SimHei')
        # plt.subplot(527), plt.imshow(edgeSegI6, 'gray'), plt.title(u'开启闭合型 I6=(f.b)-(fOb)', fontproperties='SimHei')
        # plt.subplot(528), plt.imshow(edgeSegI7, 'gray'), plt.title(u'抗噪膨胀合型 I7=(f⊕b)-(f.b)', fontproperties='SimHei')
        # plt.subplot(529), plt.imshow(edgeSegI8, 'gray'), plt.title(u'抗噪腐蚀合型 I8=(fOb)-(f⊝b)', fontproperties='SimHei')
        # plt.subplot(5, 2, 10), plt.imshow(edgeSegI9, 'gray'), plt.title(u'理想斜边缘型 I9=min{I1,I2}', fontproperties='SimHei')

        print(u' 品质因素F：')
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



